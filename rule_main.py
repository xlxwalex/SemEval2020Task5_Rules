from rules.feat_tool import feat_utils
from rules.feat_tool import Feature_Generator
from utils import read_file
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def draw_confusion(confusion_matrix):
    plt.matshow(confusion_matrix, cmap = plt.cm.Greens)
    plt.colorbar()
    for x in range(len(confusion_matrix)):
        for y in range(len(confusion_matrix)):
            plt.annotate(confusion_matrix[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

if __name__ == '__main__':
    file_path = './data/subtask1_train.txt'
    labels, sentences = read_file(file_path)
    tokens = [feat_utils.tokenize(item) for item in sentences]
    fgen = Feature_Generator()
    preds = fgen(tokens)
    print('Classification Report:')
    print(classification_report(labels, preds))
    print('Confusion Matrix:')
    conf = confusion_matrix(labels, preds)
    print(conf)
    draw_confusion(conf)