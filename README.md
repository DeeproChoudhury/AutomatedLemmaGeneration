### Run evaluations
This repository currently supports inference with both OpenAI and Mistral models.

- Please save your Mistral API key in the environment variable `MISTRAL_API_KEY`. For OpenAI, you can
    use the `OPENAI_API_KEY` environment variable, or the `OPENAI_ORGANISATION` variable for the organisation id.
- To run orientation and lemma generation on MiniF2F, run `sh runs/lemmas_test.sh` for the test set and `sh runs/lemmas_valid.sh` for the validation set.
- The shell scripts allow for modification of the choice of models, input/output/prompt directories, the temperature, timeout and type of models used. Note that as mistral and OpenAI models have different APIs, you must specify the model type in the shell script.
- The mistral models supported are `mistral-large`, `nemo` and `codestral`.