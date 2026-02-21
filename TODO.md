# TODO

- stupid backoff - use backoff weight?
- token prediction instead of char prediction - what if token is not a full char?
    - sample tokens until full char??
    - argmax? for different X, go back X characters and tokenize the string. then, consider next tokens iff they contain the suffix.
- matches are considered caseless.
- todo: non-greedy matching?
- tune vocab size