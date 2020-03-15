import os
class Logger:
    """Logger to record values to a CSV file."""

    def __init__(self, labels, path, errase=True, verbatim=True):
        self.n_labels = len(labels)
        self.path = path
        self.verbatim = verbatim
        assert errase or not os.path.exists(os.path.dirname(path)), self.path
        print('[Logger] logging {} to {}'.format(labels, self.path))
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(self.path, 'w') as csv_file:
            csv_file.write(','.join(labels) + '\n')

    def log(self, *values):
        assert len(values) == self.n_labels
        text = ','.join([str(v) for v in values])
        if self.verbatim:
            print('[Logger] {}'.format(text))
        with open(self.path, 'a') as csv_file:
            csv_file.write(text + '\n')
