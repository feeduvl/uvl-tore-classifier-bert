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


retry "timeout --foreground 1h python src/pipelines/bilstm_pipeline.py -cp experiments/ -cn 1_stage_bilstm_parameter_sweep -cd src/pipelines/conf/ -m"
retry "timeout --foreground 1h python src/pipelines/bilstm_pipeline.py -cp experiments/ -cn e2e_bilstm_parameter_sweep -cd src/pipelines/conf/ -m"





#retry "timeout --foreground 3h python src/pipelines/bert_pipeline.py -cp experiments/ -cn 1_stage_bert_parameter_sweep -cd src/pipelines/conf/ -m"

#retry "timeout --foreground 3h python src/pipelines/staged_bert_pipeline.py -cp experiments/ -cn staged_bert_parameter_sweep -cd src/pipelines/conf/ -m"

#retry "timeout --foreground 3h python src/pipelines/bert_pipeline.py -cp experiments/ -cn e2e_bert_sweep -cd src/pipelines/conf/ -m"
