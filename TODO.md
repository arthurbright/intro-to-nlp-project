# TODO

- tune vocab size
- try n = 5
- language identification?

.\.venv\Scripts\Activate.ps1
python src/myprogram.py train --work_dir work
python src/myprogram.py test --work_dir work --test_data data/open-dev/input.txt --test_output pred.txt
python grader/grade.py pred.txt.flat data/open-dev/answer.txt --verbose  

multiprocess: 4 workers: 0.064, 0.079
8 workers:  0.076
batching: 0.018
batching, 1 proc with pool: 0.011
batching, 1 proc no pool: 0.009
batching, 512 batch: 0.005

1_000_000, 20 epochs: 0.578

TODO:
- filter space-related wikipedia
- tune hyperparameters (num layers, epochs)



5-gram 50k: 0.58
5-gram 15k: 0.675
4-gram 30k: 0.674
3-gram: 0.7?? or 0.63
n = 4, 30k, byte-wise training = 500k char from each language, 4x original data upsample: 0.66
n = 4, 20k, byte-wise training = 500k char from each language, 2x upsample: 0.801
n = 4, 20k, byte-wise training = 800k char from each language, 2x upsample: 0.802
n = 4, 30k, byte-wise training = 800k char from each language, 2x upsample: 0.7942

work5 - n = 5, 20k, byte-wise training = 1000k char from each language, 2x upsample:0.803

n = 4, 10000
Success rate for en: 8601/9996 = 0.8604441776710684
Success rate for ru: 8764/9994 = 0.876926155693416
Success rate for zh: 7336/9879 = 0.7425852819111246
Success rate for ja: 8188/9958 = 0.8222534645511147
Success rate for hi: 8883/9994 = 0.888833299979988
Success rate for ar: 8593/9992 = 0.8599879903923139
Success rate for ko: 8468/9972 = 0.8491776975531489
Success rate for fr: 8777/9988 = 0.8787545054064878
Success rate for de: 8702/9992 = 0.8708967173738991
Success rate for it: 8779/9992 = 0.8786028823058447
0.85300

n = 5, 10000 
Success rate for en: 8596/9996 = 0.8599439775910365
Success rate for ru: 8764/9994 = 0.876926155693416
Success rate for zh: 7386/9879 = 0.7476465229274218
Success rate for ja: 8220/9958 = 0.8254669612371962
Success rate for hi: 8875/9994 = 0.8880328196918151
Success rate for ar: 8589/9992 = 0.8595876701361089
Success rate for ko: 8501/9972 = 0.8524869634977938
Success rate for fr: 8765/9988 = 0.8775530636764117
Success rate for de: 8696/9992 = 0.8702962369895917
Success rate for it: 8764/9992 = 0.877101681345076
0.85364

n = 5, 10000, char-wise
Success rate for en: 8599/9996 = 0.8602440976390556
Success rate for ru: 8744/9994 = 0.8749249549729838
Success rate for zh: 7441/9879 = 0.7532138880453487
Success rate for ja: 8196/9958 = 0.823056838722635
Success rate for hi: 8851/9994 = 0.8856313788272964
Success rate for ar: 8519/9992 = 0.852582065652522
Success rate for ko: 8359/9972 = 0.8382470918572001
Success rate for fr: 8772/9988 = 0.8782539046856227
Success rate for de: 8680/9992 = 0.8686949559647719
Success rate for it: 8736/9992 = 0.8742994395516414
0.85104

n = 4, 10000, char-wise
Success rate for en: 8592/9996 = 0.8595438175270108
Success rate for ru: 8737/9994 = 0.8742245347208325
Success rate for zh: 7407/9879 = 0.7497722441542666
Success rate for ja: 8161/9958 = 0.8195420767222333
Success rate for hi: 8855/9994 = 0.8860316189713828
Success rate for ar: 8525/9992 = 0.8531825460368294
Success rate for ko: 8329/9972 = 0.8352386682711592
Success rate for fr: 8783/9988 = 0.8793552262715258
Success rate for de: 8696/9992 = 0.8702962369895917
Success rate for it: 8757/9992 = 0.8764011208967174
0.85049

n = 4, 20000, truncating, char-wise
Success rate for en: 8589/9996 = 0.8592436974789915
Success rate for ru: 8748/9994 = 0.8753251951170702
Success rate for zh: 7466/9879 = 0.7557445085534973
Success rate for ja: 8258/9958 = 0.8292829885519181
Success rate for hi: 8863/9994 = 0.8868320992595558
Success rate for ar: 8557/9992 = 0.8563851080864692
Success rate for ko: 8364/9972 = 0.838748495788207
Success rate for fr: 8785/9988 = 0.8795554665598718
Success rate for de: 8719/9992 = 0.8725980784627703
Success rate for it: 8663/9992 = 0.8669935948759008
0.8521908237015948

n = 4, 20000, byte-wise
Success rate for en: 8567/9996 = 0.8570428171268507
Success rate for ru: 8688/9994 = 0.8693215929557735
Success rate for zh: 7312/9879 = 0.740155886223302
Success rate for ja: 8119/9958 = 0.8153243623217513
Success rate for hi: 8829/9994 = 0.8834300580348209
Success rate for ar: 8488/9992 = 0.8494795836669335
Success rate for ko: 8420/9972 = 0.8443642198154834
Success rate for fr: 8706/9988 = 0.8716459751702043
Success rate for de: 8598/9992 = 0.8604883907125701
Success rate for it: 8600/9992 = 0.8606885508406725
Overall Success Rate: 0.8453241376545004

gpt2-tokenizer: bad

n = 4, 30k, byte-wise training = 500k char from each language, 4x original data upsample
- intermediate results looking good
Success rate for en: 8548/9996 = 0.8551420568227291
Success rate for ru: 8629/9994 = 0.8634180508304983
Success rate for zh: 7205/9879 = 0.729324830448426
Success rate for ja: 8038/9958 = 0.8071901988351075
Success rate for hi: 8687/9994 = 0.8692215329197519
Success rate for ar: 8421/9992 = 0.8427742193755005
Success rate for ko: 8326/9972 = 0.8349378259125552
Success rate for fr: 8614/9988 = 0.8624349219062876
Success rate for de: 8567/9992 = 0.8573859087269816
Success rate for it: 8614/9992 = 0.8620896717373899
Overall Success Rate: 0.8385276221217559

^^ , char-wise
Success rate for en: 8506/9996 = 0.8509403761504601
Success rate for ru: 8515/9994 = 0.8520112067240344
Success rate for zh: 7188/9879 = 0.7276040085028849
Success rate for ja: 7960/9958 = 0.7993573006627837
Success rate for hi: 8621/9994 = 0.8626175705423254
Success rate for ar: 8402/9992 = 0.8408726981585268
Success rate for ko: 8250/9972 = 0.8273164861612515
Success rate for fr: 8586/9988 = 0.8596315578694433
Success rate for de: 8522/9992 = 0.8528823058446757
Success rate for it: 8576/9992 = 0.8582866293034428
Overall Success Rate: 0.8332848822639013
