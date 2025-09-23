import argparse
import gc
import logging
import math
import time
import torch
from transformers import AutoTokenizer

from src.evaluation import *
from src.utilities import *
from src.local_datasets import *
from src.pruning import *
from src.evopress import *
from src.slicegpt import *
from src.ablations import *


def parse_args():
  parser = argparse.ArgumentParser(description="Pruning of transformer models")
  parser.add_argument('--model', type=str, required=True, help="Specify the model's name or path to be pruned")
  parser.add_argument('--seed', type=int, default=0, help="Set a seed for reproducibility (default: 0)")
  parser.add_argument('--cache_dir', type=str, required=False, help="Path to a directory in which a downloaded pretrained model should be cached. This option is not supported when --pruning_method=slicegpt")

  parser.add_argument(
    '--dense', 
    help="Load the original dense model without pruning", 
    action='store_true'
  )

  parser.add_argument(
    '--pruning_method', 
    type=str, 
    choices=['2ssp', 'window_based', 'shortgpt', 'blockpruner', 'evopress', 'slicegpt'],
    help="Specify the pruning method to apply"
  )
  parser.add_argument(
    '--sparsity_rate', 
    type=float, 
    help="A floating-point value ranging from 0.0 to 1.0 that determines the target sparsity level for pruning. If set to -1, pruning is performed at all sparsity levels from 0.0 to 1.0 with a step size of 1/N. A value of -2 applies pruning at predefined sparsity levels of 25%%, 37.5%%, and 50%%.)"
  )

  parser.add_argument('--main_table_results', help="Generate results for the main results table in the paper (Table 1)", action='store_true')
  parser.add_argument('--evaluate_inference', help="Measure the model's inference time", action='store_true')
  parser.add_argument('--evaluate_downstream', help="Perform downstream task evaluation at 37.5%% sparsity", action='store_true')
  parser.add_argument('--evaluate_perplexity', help="Evaluates perplexity on Wikitext2 only", action='store_true')
  parser.add_argument('--evaluate_qualitative', help="Qualitative results", action='store_true')

  parser.add_argument('--local_datasets', help="Use local datasets stored in the './data/' folder", action='store_true')
  parser.add_argument(
    '--ablation', 
    action='store_true',
    help="Run the ablation study experiments"
  )
  parser.add_argument(
        '--logging',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help="Set the logging level (default: INFO)"
    )

  return parser.parse_args()


@torch.no_grad()
def main():
  args = parse_args()
  logging_level = getattr(logging, args.logging.upper())
  logging.basicConfig(
      level=logging_level,
      format='%(asctime)s - %(levelname)s - %(message)s',
      datefmt='%H:%M:%S'
  )
  
  set_seed(args.seed)
  
  # Load the tokenizer
  logging.info("Loading the tokenizer")
  tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
  logging.info("Loaded the tokenizer")

  ###################### Datasets 
  logging.info("Loading the Datasets")
    
  # Evaluation datasets
  dataset_wikitext = load_wikitext2(args.local_datasets)
  dataset_c4_val = load_c4(train=False, local=args.local_datasets)
  dataset_fineweb_edu = load_fineweb_edu(local=args.local_datasets)[:500]
  dataset_c4_train = load_c4(train=True, local=args.local_datasets)
  logging.info("Loaded the Datasets")

  logging.info("Tokenizing the Datasets")
  wikitext_input_ids = tokenizer("\n\n".join(dataset_wikitext["text"]), return_tensors="pt", add_special_tokens=False).input_ids
  c4_val_input_ids = tokenizer("\n\n".join(dataset_c4_val["text"]), return_tensors="pt", add_special_tokens=False).input_ids
  fineweb_edu_input_ids = tokenizer("\n\n".join(dataset_fineweb_edu["text"]), return_tensors="pt", add_special_tokens=False).input_ids
  logging.info("Tokenized the datasets")

  # Calibration datasets
  num_calibration_samples_2ssp = 32
  num_calibration_samples = 256 # For SliceGPT, ShortGPT and Window Based

  calibration_dataset = get_calibration(dataset_c4_train, tokenizer, num_samples=num_calibration_samples, seq_len=2048)

  calibration_dataset_2ssp = calibration_dataset[:num_calibration_samples_2ssp]
  first_calibration_sample = calibration_dataset[0]
  

  ###################### Dense model
  if args.dense:

    logging.info("Dense model evaluation")
    logging.info("Loading the model")    
    model = loadModel(args.model, args.cache_dir)
    logging.debug(model)
    printModelStats(model, "Dense model")
    
    if args.evaluate_inference == True:
      evaluate_inference_time(model, first_calibration_sample)

    if args.evaluate_downstream == True:
      evaluation_downstream(model, args.model)

    if args.main_table_results == True:
      evaluation_ppl(model, wikitext_input_ids, c4_val_input_ids, fineweb_edu_input_ids)
    
    if args.evaluate_perplexity == True:
      ppl = evaluate_perplexity(model, wikitext_input_ids, seq_len=2048)
      logging.info(f"Perplexity (wikitext2): {ppl}")  

    if args.evaluate_qualitative == True:
      qualitative_results(model, tokenizer, max_length=128)
    

  ###################### Pruning

  pruning_method = args.pruning_method
  if pruning_method is not None:

    sparsity_rate = args.sparsity_rate

    logging.info("Loading the model")
    model = loadModel(args.model, args.cache_dir)
    num_blocks = len(model.model.layers)
    logging.debug(model)

    if pruning_method == "slicegpt":
      del model # the model will be loaded by the SliceGPT model adapter
      gc.collect()
      torch.cuda.empty_cache()
      

    if int(round(sparsity_rate)) == -1: # prune all possible blocks
      pruning_rates = [i / num_blocks for i in range(1, num_blocks - 1)]
    elif int(round(sparsity_rate)) == -2: # prune at 25%, 37.5%, 50%
      pruning_rates = [0.25, 0.375, 0.5]
    else: # Prune a single sparsity rate
      pruning_rates = [sparsity_rate]

    for target_sparsity in pruning_rates:

      set_seed(args.seed)

      # Measure pruning time
      start_time = time.time()

      if pruning_method in ["window_based", "shortgpt", "blockpruner", "evopress"]:
        target_sparsity_blocks = target_sparsity * num_blocks
        if not target_sparsity_blocks.is_integer():
          logging.warning(f"Invalid sparsity rate for {pruning_method}: must be a multiple of 1/{num_blocks} since model has {num_blocks} blocks.")
          target_sparsity_blocks = int(math.ceil(target_sparsity_blocks)) 
          logging.warning(f"Rounding to next valid sparsity rate: {target_sparsity_blocks/num_blocks:.6f} ({int(target_sparsity_blocks)} blocks)")
        else:
          target_sparsity_blocks = int(target_sparsity_blocks)
        
        target_sparsity = target_sparsity_blocks / num_blocks

      logging.info(f"Pruning rate {target_sparsity*100} (equivalent of {target_sparsity * num_blocks} blocks)")


      attnMask = mlpMask = None
      if pruning_method == "window_based":
        attnMask = mlpMask = window_based(model, target_sparsity_blocks, calibration_dataset)
      elif pruning_method == "shortgpt":
        attnMask = mlpMask = shortGPT(model, target_sparsity_blocks, calibration_dataset)
      elif pruning_method == "blockpruner":
        attnMask, mlpMask = blockpruner(model, target_sparsity_blocks, first_calibration_sample)
      elif pruning_method == "evopress":
        attnMask, mlpMask = evopress(model, target_sparsity_blocks, tokenizer, dataset_c4_train, drop_entire_block=False)
      elif pruning_method == "2ssp":
        model = two_stage_2ssp(model, calibration_dataset_2ssp, target_sparsity)
      elif pruning_method == "slicegpt":
        model = slicegpt(args.model, target_sparsity, calibration_dataset)
      else:
        logging.error("Invalid method provided")
        exit(1)

      end_time = time.time()
      logging.info(f"Pruning Time: {end_time - start_time} s")

      printModelStats(model, "Pruned model")
      
      # 2SSP: no masks are generated. The parameters are removed
      if attnMask is None:
        
        if args.evaluate_inference == True:
          evaluate_inference_time(model, first_calibration_sample)
        
        if args.evaluate_downstream == True:
          evaluation_downstream(model, args.model)

        if args.main_table_results == True:
          evaluation_ppl(model, wikitext_input_ids, c4_val_input_ids, fineweb_edu_input_ids)         
        
        if args.evaluate_perplexity == True:
          ppl = evaluate_perplexity(model, wikitext_input_ids, seq_len=2048)
          logging.info(f"Perplexity (wikitext2): {ppl}")       

        if args.evaluate_qualitative == True:
          qualitative_results(model, tokenizer, max_length=128)

        reset_mlps_shape(model)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if pruning_method != "slicegpt":
          model = loadModel(args.model, args.cache_dir)

      else:
        logging.debug(f"Pruned blocks: attn={attnMask} mlp={mlpMask}")

        maskModel(model, attnMask=attnMask, mlpMask=mlpMask)

        if args.evaluate_inference == True:
          evaluate_inference_time(model, first_calibration_sample)

        if args.evaluate_downstream == True:
          evaluation_downstream(model, args.model)

        if args.main_table_results == True:
          evaluation_ppl(model, wikitext_input_ids, c4_val_input_ids, fineweb_edu_input_ids)
        
        if args.evaluate_perplexity == True:
          ppl = evaluate_perplexity(model, wikitext_input_ids, seq_len=2048)
          logging.info(f"Perplexity (wikitext2): {ppl}")

        if args.evaluate_qualitative == True:
          qualitative_results(model, tokenizer, max_length=128)
        
        unmaskModel(model, attnMask=attnMask, mlpMask=mlpMask)

  ###################### Ablations
  if args.ablation:
    run_ablations(args, tokenizer, dataset_c4_train, wikitext_input_ids, calibration_dataset_2ssp)
    

if __name__ == "__main__":
  main()
