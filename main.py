from PySide6.QtWidgets import QApplication
import ConnectFour_GUI
import sys

if __name__ == '__main__':
    if QApplication.instance():
        app = QApplication.instance()
    else:
        app = QApplication([])
    gameMainWindow = ConnectFour_GUI.mainWindow()
    def on_app_exit():
        gameMainWindow.alphazero.terminate()
        gameMainWindow.timer.timer.stop()
        gameMainWindow.timer.quit()
        gameMainWindow.alphazero.quit()
    app.aboutToQuit.connect(on_app_exit)
    sys.exit(app.exec())
