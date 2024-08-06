import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, LlamaForCausalLM, AutoConfig
from rtfm.configs import TrainConfig, TokenizerConfig
from rtfm.inference_utils import InferenceModel
from rtfm.serialization.serializers import get_serializer
from rtfm.tokenization.text import prepare_tokenizer
from datasets import load_dataset
import random

def setup_model():

    train_config = TrainConfig(model_name="mlfoundations/tabula-8b", context_length=8192)

    # If using a base llama model (not fine-tuned TabuLa),
    # make sure to set add_serializer_tokens=False
    # (because we do not want to use special tokens for 
    # the base model which is not trained on them).
    tokenizer_config = TokenizerConfig()

    # Load the configuration
    config = AutoConfig.from_pretrained(train_config.model_name)

    # Set the torch_dtype to bfloat16 which matches TabuLa train/eval setup
    config.torch_dtype = 'bfloat16'

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LlamaForCausalLM.from_pretrained(
        train_config.model_name, device_map="auto", config=config).to(device)

    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    serializer = get_serializer(train_config.serializer_cls)

    tokenizer, model = prepare_tokenizer(
        model,
        tokenizer=tokenizer,
        pretrained_model_name_or_path=train_config.model_name,
        model_max_length=train_config.context_length,
        use_fast_tokenizer=tokenizer_config.use_fast_tokenizer,
        serializer_tokens_embed_fn=tokenizer_config.serializer_tokens_embed_fn,
        serializer_tokens=serializer.special_tokens
        if tokenizer_config.add_serializer_tokens
        else None,
    )

    return InferenceModel(model=model, tokenizer=tokenizer, serializer=serializer)

def discretize_continuous_column(column: pd.Series, thresholds) -> pd.Series:
    """Take a continuous-valued column and discretize it into num_buckets.

    The formatting of the outputs is of the form 'less than 0', 'between 0.5 and 0.9', or 'greater than 1.99'
    etc. depending on the value of the observation and the number of buckets used.

    This is the same format used in training TabuLa-8B and should be used for inference and
    evaluation of that model.
    """
    assert pd.api.types.is_numeric_dtype(column)

    # Compute bucket thresholds
    # thresholds = [column.quantile(i / num_buckets) for i in range(1, num_buckets)]
    

    # Define a function to categorize each value
    def categorize_value(x):
        for i, threshold in enumerate(thresholds):
            if x < threshold:
                if i == 0:
                    return f"less than {threshold}"
                else:
                    return f"between {thresholds[i-1]} and {threshold}"
        return f"greater than {thresholds[-1]}"

    # Apply the categorization function to the column
    return column.apply(categorize_value)

def get_regression_bucket_choices(thresholds):
    target_choices = []
    for i, threshold in enumerate(thresholds):
        if i == 0:
            target_choices.append(f"less than {threshold}") 
        else:
            target_choices.append(f"between {thresholds[i-1]} and {threshold}")
    target_choices.append(f"greater than {thresholds[-1]}")
    return target_choices

def regress_example(model, df, target_col, target_ix, shots_ixs, lower_bound, upper_bound, num_buckets, tol=.01):

    while upper_bound - lower_bound >  tol:
        """
        this logic is a bit confusing, but the following example helps
        say the bound in [0,1] and you have 3 buckets
        you want to do
        less than .333, betweeen .333 and .666, and greater than .666
        so len(thresholds) = num_buckets - 1
        and the thresholds are always strictly between the upper and lower bounds
        """
        thresholds = np.linspace(lower_bound, upper_bound, num_buckets + 1)[1:-1].round(4)

        curr_df = df.copy()

        curr_df[target_col] = discretize_continuous_column(curr_df[target_col], thresholds)
        target_choices = get_regression_bucket_choices(thresholds)

        output = model.predict(
            target_example = curr_df.iloc[[target_ix]],
            target_colname = target_col,
            target_choices = target_choices,
            max_new_tokens = 50,
            labeled_examples = curr_df.iloc[shots_ixs],
        )
        # print(lower_bound, upper_bound)
        # print(thresholds)
        # print(target_choices)
        # print(output)
        # print("------- \n")

        num_thresholds = len(target_choices) 

        i = target_choices.index(output) # returns index of matching output, Value error if not

        if i == 0:
            upper_bound = thresholds[0]
        elif i > 0 and i == num_thresholds - 1:
            lower_bound = thresholds[-1]
        else:
            lower_bound = thresholds[i-1]
            upper_bound = thresholds[i]

    return (upper_bound + lower_bound) / 2


if __name__ == "__main__":
    dataset = load_dataset("inria-soda/tabular-benchmark", data_files="reg_cat/house_sales.csv")
    dataset = dataset['train'].to_pandas().sample(frac=1)

    inference_model = setup_model()

    df = dataset.round(4)
    TARGET_COL = df.columns[-1]
    N = 500
    init_a = df[TARGET_COL].min()
    init_b = df[TARGET_COL].max()
    num_buckets = 4
    # TARGET_COL, init_a, init_b, num_buckets
    range_shots = [4, 8, 16]

    res = []
    for i in range(N):
        for num_shots in range_shots:
            available_ixs = list(range(df.shape[0]))
            del available_ixs[i]
            shot_ixs = random.choices(available_ixs, k=num_shots)

            pred = regress_example(inference_model, df, TARGET_COL, i, shot_ixs, 
                                   init_a, init_b, num_buckets, tol=.01)

            res.append((i, df.iloc[i][TARGET_COL], pred, num_shots))
    
    res_df = pd.DataFrame(res, columns=['ix', 'true_label', 'prediction', 'num_shots'])
    res_df.to_csv('house_sales_buckets_{}.csv'.format(num_buckets), index=False)
    
