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



retry "timeout --foreground 2h python src/experiments/dual_model_staged_bert.py -cp configurations/ -cn 2_stage_dual_model_bert_bert_sweep -cd src/experiments/conf/ -m"
