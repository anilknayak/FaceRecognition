from pyqtgraph.Qt import QtCore, QtGui
import sys


class MenuFrame():
    def __init__(self, mainWindow):
        self.main = mainWindow
        # main_menu = mainWindow.menuBar()
        self.createActions()
        self.createMenus()

        # file_menu = main_menu.addMenu('&File')
        # quitAction = QtGui.QAction("& Quit", mainWindow)
        # quitAction.setShortcut("Ctrl+Q")
        # quitAction.setStatusTip('Closing The Application')
        # quitAction.triggered.connect(self.close_application)
        # file_menu.addAction(quitAction)
        #
        # setting_menu = main_menu.addMenu('& Setting')
        # configuratonAction = QtGui.QAction("& Configuration", mainWindow)
        # configuratonAction.setShortcut("Ctrl+C")
        # configuratonAction.setStatusTip('Opening Configuration')
        # configuratonAction.triggered.connect(self.configuraton_action)
        # setting_menu.addAction(configuratonAction)
        #
        # help_menu = main_menu.addMenu('& Help')
        # aboutAction = QtGui.QAction("& About", mainWindow)
        # aboutAction.setShortcut("Ctrl+A")
        # aboutAction.setStatusTip('Opening About')
        # aboutAction.triggered.connect(self.about_action)
        # help_menu.addAction(aboutAction)
        # helpAction = QtGui.QAction("& Help", mainWindow)
        # helpAction.setShortcut("Ctrl+H")
        # helpAction.setStatusTip('Opening Help')
        # helpAction.triggered.connect(self.about_action)
        # help_menu.addAction(helpAction)

    def createActions(self):
        self.newAct = QtGui.QAction("& New", self.main, shortcut="Ctrl+N",
                                     statusTip="New the application", triggered=self.new)
        self.connectAct = QtGui.QAction("& Connect",  self.main, shortcut="Ctrl+C",
                                     statusTip="Connect the application", triggered=self.connect)
        self.quitAct = QtGui.QAction("& Quit",  self.main, shortcut="Ctrl+Q",
                                     statusTip="Quit the application", triggered=self.close)

        self.settingAct = QtGui.QAction("& Setting",  self.main, shortcut="Ctrl+S",
                                     statusTip="Setting the application", triggered=self.setting)

        self.aboutAct = QtGui.QAction("&About",  self.main,
                                      statusTip="Show the application's About box",
                                      triggered=self.about)

    def createMenus(self):
        self.main.fileMenu = self.main.menuBar().addMenu("&File")
        self.main.fileMenu.addAction(self.newAct)
        self.main.fileMenu.addAction(self.connectAct)
        self.main.fileMenu.addSeparator()
        self.main.fileMenu.addAction(self.quitAct)

        self.main.editMenu = self.main.menuBar().addMenu("&Setting")

        self.main.viewMenu = self.main.menuBar().addMenu("&View")

        self.main.menuBar().addSeparator()

        self.main.reporting = self.main.menuBar().addMenu("&Reporting")

        self.main.menuBar().addSeparator()

        self.main.helpMenu = self.main.menuBar().addMenu("&Help")
        self.main.helpMenu.addAction(self.aboutAct)

    def close(self):
        self.main.exit()

    def setting(self):
        print('setting')

    def about(self):
        print('About')

    def new(self):
        print('new')

    def connect(self):
        print('connect')
