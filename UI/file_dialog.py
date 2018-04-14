import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from pyqtgraph.Qt import QtGui

class OpenFileDialog(QWidget):
    def __init__(self):
        super().__init__()
        screen = QtGui.QDesktopWidget().availableGeometry()
        self.screen_width = screen.width()
        self.screen_height = screen.height()
        self.title = 'Choose Video Files'
        self.left = self.screen_width // 2 - 300
        self.top = self.screen_height // 2 - 200
        self.width = 640
        self.height = 480

        self.filename = None


    def initUI(self, type):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        if type == 'video':
            self.openFileNameDialog()
        elif type == 'pickle':
            self.openFileNamesDialog_Pickle()
        # self.openFileNamesDialog()
        # self.saveFileDialog()

        self.show()

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.filename, _ = QFileDialog.getOpenFileName(self, "Select Video File", "",
                                                  "AVI (*.avi);; MOV (*.MOV);; MP4 (*.mp4);; MPEG(*.mpeg)", options=options)

    def selectDirectory(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.directory_name = QFileDialog.getExistingDirectory(self, "Select Directory")
        self.show()

    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "QFileDialog.getOpenFileNames()", "",
                                                "All Files (*);;Python Files (*.py)", options=options)
        if files:
            print(files)

    def openFileNamesDialog_Pickle(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "QFileDialog.getOpenFileNames()", "",
                                                "All Files (*);;Pickle (*.pickle);;Pkl (*.pkl)", options=options)
        if files:
            print(files)

    def saveFileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(self, "QFileDialog.getSaveFileName()", "",
                                                  "All Files (*);;Text Files (*.txt)", options=options)
        if fileName:
            print(fileName)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = OpenFileDialog()
    sys.exit(app.exec_())