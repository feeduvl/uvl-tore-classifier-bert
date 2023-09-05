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



retry "timeout --foreground 2h python src/experiments/fake_staged_bert.py -cp configurations/ -cn fake_staged_bert -cd src/experiments/conf/ "
