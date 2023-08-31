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


retry "timeout --foreground 1h python src/experiments/sner.py -cp configurations/ -cn 1_stage_sner_parameter_sweep -cd src/experiments/conf/ -m"
retry "timeout --foreground 1h python src/experiments/sner.py -cp configurations/ -cn e2e_sner_parameter_sweep -cd src/experiments/conf/ -m"
