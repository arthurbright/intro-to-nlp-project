# TODO

- speed it up
    - current approach uses pruning: only use tokens taht show up, and only consider a token X to follow Y if (Y, X) actually showed up in the corpus
- todo: non-greedy matching?
- tune vocab size
- deal EMPTY GUESSES, likely due to no valid tokenization or only valid tokenization using un-used tokens?
    - maybe more data 
    - instead of encoding entire prefix, just try last few chars?

3 1991
1 2909
2 846
-1 137
11 64
12 44
-2 9


1: 200
2: 50
2909 + 846 * 200 + 1991 * (200 * 200 + 50)


empty preds: Prediction 33268: []
0
Prediction 33269: []

juri al
jura,
EMPTY GUESSES: 좋습니다, 이것은 o<sub>2</sub> 고속 모드에서 3분 10초입니다. 5분 동안 유지하고 싶지 않습니다. 이는 사람이 견딜 수 있는 것보다 많다고 생각합니다. 슈트 입구 온도가 64로
EMPTY GUESSES: 좋습니다, 이것은 o<sub>2</sub> 고속 모드에서 3분 10초입니다. 5분 동안 유지하고 싶지 않습니다. 이는 사람이 견딜
Prediction 31400: ['는', '요', ' ']
Prediction 31600: ['다', '까', ' ']
EMPTY GUESSES: 호라이즌 스캐너 점검을 해야 하
EMPTY GUESSES: 호라이즌 스캐너 점
EMPTY GUESSES: यह कहता है कि गेंद पर माइनस 10° नीचे है। मुझे लगता है कि यह क
EMPTY GUESSES: यह कहता है कि गेंद पर माइनस 10° नीचे है। मुझे लगता है कि यह कक्षा दर है, ह
i've been advised from the cape you might put your prop switch off 