from lstm_network import create

neural_net = create()


def get_sentiment(sentence):
    prediction = neural_net.predict(sentence)
    negative_score = prediction[0]
    non_negative_score = prediction[1]
    print(f'Positive: {non_negative_score}\n'
          f'Negative: {negative_score}\n'
          f'Composite: {non_negative_score - negative_score}')
    return non_negative_score - negative_score


def test():
    neural_net.test_model()


if __name__ == '__main__':
    test()

    while True:
        get_sentiment(input('Input: '))
