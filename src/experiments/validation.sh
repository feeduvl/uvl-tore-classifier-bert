retry() {

  local command="$1" # Second argument

  # Run the command, and save the exit code
  $command
  local exit_code=$?

  # If the exit code is non-zero (i.e. command failed), and we have not
  # reached the maximum number of retries, run the command again
  if [[ $exit_code -ne 0 ]]; then
    retry "$command"
  else
    # Return the exit code from the command
    return $exit_code
  fi
}



retry "timeout --foreground 1h python src/experiments/sner.py -cp configurations/ -cn 1_stage_sner_validation -cd src/experiments/conf/ "
retry "timeout --foreground 1h python src/experiments/bilstm.py -cp configurations/ -cn 1_stage_bilstm_validation -cd src/experiments/conf/ "
retry "timeout --foreground 2h python src/experiments/bert.py -cp configurations/ -cn 1_stage_bert_validation -cd src/experiments/conf/ -m"

retry "timeout --foreground 1h python src/experiments/sner.py -cp configurations/ -cn e2e_sner_validation -cd src/experiments/conf/ "
retry "timeout --foreground 1h python src/experiments/bilstm.py -cp configurations/ -cn e2e_bilstm_validation -cd src/experiments/conf/"
retry "timeout --foreground 2h python src/experiments/bert.py -cp configurations/ -cn e2e_bert_validation -cd src/experiments/conf/ -m"

retry "timeout --foreground 2h python src/experiments/dual_model_staged_bert.py -cp configurations/ -cn 2_stage_dual_model_bert_bert_validation_outlier -cd src/experiments/conf/ -m"
retry "timeout --foreground 2h python src/experiments/dual_model_staged_bert.py -cp configurations/ -cn 2_stage_dual_model_bert_bert_validation_selected -cd src/experiments/conf/ -m"

retry "timeout --foreground 2h python src/experiments/fake_staged_bert.py -cp configurations/ -cn fake_staged_bert -cd src/experiments/conf/ -m"

retry "timeout --foreground 2h python src/experiments/bert.py -cp configurations/ -cn 1_stage_bert_validation_high_recall -cd src/experiments/conf/ -m"
retry "timeout --foreground 2h python src/experiments/bert.py -cp configurations/ -cn 1_stage_bert_validation_high_precision -cd src/experiments/conf/ -m"


retry "timeout --foreground 2h python src/experiments/dual_model_staged_bert.py -cp configurations/ -cn 2_stage_dual_model_bert_bert_high_recall -cd src/experiments/conf/ -m"
retry "timeout --foreground 2h python src/experiments/dual_model_staged_bert.py -cp configurations/ -cn 2_stage_dual_model_bert_bert_high_precision -cd src/experiments/conf/ -m"