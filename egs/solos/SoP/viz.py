import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_loss_metrics(path, history):
    fig = plt.figure()
    plt.plot(history['train']['epoch'], history['train']['err'],
             color='b', label='training')
    plt.plot(history['val']['epoch'], history['val']['err'],
             color='c', label='validation')
    plt.legend()
    fig.savefig(os.path.join(path, 'loss.png'), dpi=200)
    plt.close('all')

    fig = plt.figure()
    plt.plot(history['val']['epoch'], history['val']['sdr'],
             color='r', label='SDR')
    plt.plot(history['val']['epoch'], history['val']['sir'],
             color='g', label='SIR')
    plt.plot(history['val']['epoch'], history['val']['sar'],
             color='b', label='SAR')
    plt.legend()
    fig.savefig(os.path.join(path, 'metrics.png'), dpi=200)
    plt.close('all')


class HTMLVisualizer():
    def __init__(self, fn_html):
        self.fn_html = fn_html
        self.content = '<table>'
        self.content += '<style> table, th, td {border: 1px solid black;} </style>'

    def add_header(self, elements):
        self.content += '<tr>'
        for element in elements:
            self.content += '<th>{}</th>'.format(element)
        self.content += '</tr>'

    def add_rows(self, rows):
        for row in rows:
            self.add_row(row)

    def add_row(self, elements):
        self.content += '<tr>'

        # a list of cells
        for element in elements:
            self.content += '<td>'

            # fill a cell
            for key, val in element.items():
                if key == 'text':
                    self.content += val
                elif key == 'image':
                    self.content += '<img src="{}" style="max-height:256px;max-width:256px;">'.format(val)
                elif key == 'audio':
                    self.content += '<audio controls><source src="{}"></audio>'.format(val)
                elif key == 'video':
                    self.content += '<video src="{}" controls="controls" style="max-height:256px;max-width:256px;">'.format(val)
            self.content += '</td>'

        self.content += '</tr>'

    def write_html(self):
        self.content += '</table>'
        with open(self.fn_html, 'w') as f:
            f.write(self.content)
