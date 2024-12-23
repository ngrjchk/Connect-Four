import ConnectFour_AlphaZero_agent
import ConnectFour_Logic
from enum import Enum
import random
import time
import functools
import numpy as np
import torch
from copy import copy
from PySide6.QtCore import QTimer, Qt, Signal, QRect, QThread, Slot
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QBrush
from PySide6.QtWidgets import QWidget, QPushButton, QMainWindow, QTextBrowser, QGraphicsBlurEffect

class CurrentPlayer(Enum):
    HUMAN = 1
    NONE = 0
    COMPUTER = -1

class gameStatus(Enum):
    NOT_STARTED = 0
    HUMAN_LOST = 1
    DRAW = 2
    HUMAN_WON = 3
    IN_PROGRESS = 4
    
class AlphaZero(QThread):
    def __init__(self, parent):
        super().__init__(parent)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.game = ConnectFour_Logic.ConnectFour()
        self.args = {
            'C': 2,
            'num_searches': 4096,
            'dirichlet_epsilon': 0.0,
            'dirichlet_alpha': 0.3
        }
        self.model = ConnectFour_AlphaZero_agent.ResNet(self.game, 9, 128, self.device)
        self.model.load_state_dict(torch.load("./ConnectFour_Weights.pt", map_location=self.device))
        self.model.eval()
        self.mcts = ConnectFour_AlphaZero_agent.MCTS(self.game, self.args, self.model)
        self._running = True

    def run(self):
        state = copy(self.parent().board.state)
        player = copy(self.parent().player.value)
        if self.parent().status == gameStatus.IN_PROGRESS and self.parent().player == CurrentPlayer.COMPUTER:
            neutral_state = self.game.change_perspective(copy(state), player)
            mcts_probs = self.mcts.search(neutral_state)
            action = np.argmax(mcts_probs)
            if state[1, action] != 0:
                self.parent().board.button_i_status[action] = False
            next_state = self.game.get_next_state(state, action, player)
            val, is_terminal = self.game.get_value_and_terminated(next_state, action)
            if self._running == False:
                return 0
            else:
                return_dict = {"state": next_state, "val": val, "is_terminal": is_terminal}
                self.parent().computerMoveSignal.emit(return_dict)

class gameInfoWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.currentPiece = ["🔵", "🟢"]
        self.turnStr = ["My Turn", "Your Turn", "My Turn"]

    def draw_status(self, qp):
        self.emoji_font = QFont("HP Simplified JPan", 50)
        self.emoji_rect = QRect(
            self.rect().x(),
            self.rect().y(),
            self.rect().width(),
            self.rect().height()*(2/3)
        )
        self.status_text_font = QFont("HP Simplified JPan", 25)
        self.status_text_rect = QRect(
            self.rect().x(),
            self.rect().height()*(2/3),
            self.rect().width(),
            self.rect().height()*(1/3)
        )
        if self.parent().status != gameStatus.IN_PROGRESS:
            if self.parent().status.value == 0:
                qp.setFont(self.emoji_font)
                qp.drawText(
                    self.emoji_rect,
                    Qt.AlignCenter,
                    "😴"
                )
                qp.setFont(self.status_text_font)
                qp.drawText(
                    self.status_text_rect,
                    Qt.AlignCenter,
                    "Not Started"
                )
            elif self.parent().status.value == 1:
                qp.setFont(self.emoji_font)
                qp.drawText(
                    self.emoji_rect,
                    Qt.AlignCenter,
                    "👍"
                )
                qp.setFont(self.status_text_font)
                qp.drawText(
                    self.status_text_rect,
                    Qt.AlignCenter,
                    "You Lost"
                )
            elif self.parent().status.value == 2:
                qp.setFont(self.emoji_font)
                qp.drawText(
                    self.emoji_rect,
                    Qt.AlignCenter,
                    "😅"
                )
                qp.setFont(self.status_text_font)
                qp.drawText(
                    self.status_text_rect,
                    Qt.AlignCenter,
                    "phew, Draw!"
                )
            else:
                qp.setFont(self.emoji_font)
                qp.drawText(
                    self.emoji_rect,
                    Qt.AlignCenter,
                    "👏"
                )
                qp.setFont(self.status_text_font)
                qp.drawText(
                    self.status_text_rect,
                    Qt.AlignCenter,
                    "You Won!"
                )
        else:
            qp.setFont(self.emoji_font)
            qp.drawText(
                self.emoji_rect,
                Qt.AlignCenter,
                f"{self.currentPiece[self.parent().board.moveCount%2-1]}"
            )
            qp.setFont(self.status_text_font)
            qp.drawText(
                self.status_text_rect,
                Qt.AlignCenter,
                f"{self.turnStr[self.parent().player.value]}"
            )
            
    def paintEvent(self, event):
        qp = QPainter(self)
        if not qp.isActive():
            qp.begin(self)
        qp.setPen(Qt.white)
        qp.fillRect(self.rect(), QBrush(QColor(0, 0, 0)))
        self.draw_status(qp)
        qp.end()
            
class Timer(QThread):
    def __init__(self, parent):
        super().__init__(parent)
        self.timer = QTimer(self)
        self.max_time_ms = 60 * 1000
        self.time_elapsed = 0
        self.timer.timeout.connect(self.updateTimer)

    def updateTimer(self):
        self.time_elapsed += 100
        if self.time_elapsed > self.max_time_ms:
            self.timer.stop()
            self.parent().alphazero._running = False
            if self.parent().player.value == 1:
                self.parent().status = gameStatus(1)
            else:
                self.parent().status = gameStatus(3)
            self.parent().gameOverSignal.emit()
        self.parent().timeElapsedSignal.emit(self.time_elapsed)
    
    def resetTimer(self):
        self.time_elapsed = 0
        self.parent().timeElapsedSignal.emit(self.time_elapsed)
        if self.parent().status != gameStatus.IN_PROGRESS:
            self.timer.stop()

class TimerWidget(QWidget):
    def __init__(self, parent, cellWidth):
        super().__init__(parent)
        self.cellWidth = cellWidth
        self.max_time_ms = self.parent().timer.max_time_ms
        self.time_elapsed = 0

    def timeElapsed(self, time_elapsed):
        self.time_elapsed = time_elapsed
        self.update()

    def paintEvent(self, event):
        qp = QPainter(self)
        if not qp.isActive():
            qp.begin(self)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.fillRect(self.rect(), QBrush(QColor(0, 0, 0)))
        rect = self.rect()
        w = min(rect.width(), rect.height()) - self.cellWidth
        rect = QRect(self.cellWidth/2, self.cellWidth/2, w, w)
        start_angle = 90 * 16 
        span_angle = abs(int((self.max_time_ms - self.time_elapsed) / self.max_time_ms * 360 * 16))
        pen = QPen()
        pen.setColor(QColor(255, 255, 255))
        pen.setWidth(10)
        qp.setPen(pen)
        qp.drawEllipse(rect)
        if self.parent().board.moveCount%2 == 0:
            pen.setColor(QColor(0, 210, 106))
        else:
            pen.setColor(QColor(0, 116, 186))
        pen.setWidth(11)
        qp.setPen(pen)
        qp.drawArc(rect, start_angle, span_angle)
        font = QFont()
        font.setPointSize(14)
        qp.setFont(font)
        time_left_in_seconds = int(abs(self.max_time_ms - self.time_elapsed)/1000)
        qp.drawText(rect, Qt.AlignCenter, f"Time Left:\n{time_left_in_seconds} seconds")
        qp.end()

class Connect4Board(QWidget):
    def __init__(self, parent, cellWidth):
        super().__init__(parent)
        self.cellWidth = cellWidth
        self.c4logic = ConnectFour_Logic.ConnectFour()
        self.buttons = []
        self.button_i_status = [True for _ in range(7)]
        self.state = np.zeros([6,7])
        self.moveSequence = np.zeros([6,7])
        self.moveCount = 0
        self.initUI()

    def initUI(self):
        self.setGeometry(self.rect())
        self.createButtons()
        self.show()

    def createButtons(self):
        font = QFont("HP Simplified JPan", 15) 
        for column in range(7):
            self.buttons.append(QPushButton(f"{column+1}", self))
            self.buttons[column].setFont(font)
            self.buttons[column].setGeometry((column+0.75)*self.cellWidth, 6.75*self.cellWidth, self.cellWidth/2, self.cellWidth/2)
            self.buttons[column].clicked.connect(functools.partial(self.buttonClicked, column))
        if self.parent().player == CurrentPlayer.COMPUTER:
            self.toggleMoveButtons(False)

    def toggleMoveButtons(self, on=False):
        if on:
            for column in range(0, 7):
                if self.button_i_status[column]:
                    self.buttons[column].setEnabled(True)
        else:
            for column in range(0, 7):
                self.buttons[column].setEnabled(False)

    def paintEvent(self, event):
        qp = QPainter(self)
        if not qp.isActive():
            qp.begin(self)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.fillRect(self.rect(), QBrush(QColor(0, 0, 0)))
        self.drawGrid(qp)
        self.drawMoves(qp)
        qp.end()
        
    def drawGrid(self, qp):
        qp.setPen(QColor(150, 150, 150))
        qp.setBrush(Qt.white)
        for i in range(0, 7):
           qp.drawLine(0.48*self.cellWidth, (i+0.5)*self.cellWidth, 7.48*self.cellWidth, (i+0.5)*self.cellWidth)
        for i in range(0, 8):
           qp.drawLine((i+0.48)*self.cellWidth, self.cellWidth/2, (i+0.48)*self.cellWidth, 6.5*self.cellWidth)

    def drawMoves(self, qp):
        for i in range(0, 6):
            for j in range(0, 7):
                if self.state[i][j]:
                    if abs(self.moveSequence[i][j])%2 == 0:
                        qp.setPen(QColor(0, 210, 106))
                        qp.setBrush(QColor(0, 210, 106))
                    else:
                        qp.setPen(QColor(0, 116, 186))
                        qp.setBrush(QColor(0, 116, 186))
                    qp.drawEllipse((j+0.58)*self.cellWidth, (i+0.6)*self.cellWidth, 0.8*self.cellWidth, 0.8*self.cellWidth)
            
    def buttonClicked(self, column):
        row = np.max(np.where(self.state[:, column] == 0))
        if row == 0:
            self.button_i_status[column] = False
        self.toggleMoveButtons(False)
        self.state[row][column] = self.parent().player.value
        self.moveSequence[row][column] = self.moveCount*self.parent().player.value
        self.update()
        if self.c4logic.check_win(self.state, column):
            self.parent().status = gameStatus.HUMAN_WON
            self.parent().gameOverSignal.emit()
            print(self.parent().player,"Won")
        elif np.sum(self.c4logic.get_valid_moves(self.state)) == 0:
            self.parent().status = gameStatus.DRAW
            self.parent().gameOverSignal.emit()
            print("Draw")
        else:
            self.moveCount += 1
            self.parent().timer.resetTimer()
            self.parent().switchPlayer()
            self.parent().undoMoveButton.undoMoveButton.setEnabled(True)
            self.parent().alphazero._running = True
            self.parent().alphazero.start()
        self.parent().gameInfo.update()

    def computerMove(self, data):
        lonely_next_move = np.zeros([6, 7])
        for row in range(6):
            for col in range(7):
                lonely_next_move[row][col] = self.state[row][col] - data["state"][row][col]
        for row in range(6):
            for column in range(7):
                if lonely_next_move[row][column] == 1:
                    self.state[row][column] = -1
                    self.moveSequence[row][column] = self.moveCount*self.parent().player.value
                    break
            if lonely_next_move[row][column] == 1:
                break
        val = data["val"]
        self.update()
        if data["is_terminal"]:
            if val == 1:
                self.parent().status = gameStatus.HUMAN_LOST
                print(self.parent().player,"Won")
            else:
                self.parent().status = gameStatus.DRAW
                print("Draw")
            self.parent().gameOverSignal.emit()
        else:
            self.moveCount += 1
            if not self.parent().pauseGame.pauseFlag:
                self.toggleMoveButtons(True)
            self.parent().timer.resetTimer()
            self.parent().switchPlayer()
        self.parent().gameInfo.update()

class pieceSelector(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.text_browser = QTextBrowser(self)
        self.text_browser.setStyleSheet("border: none; background: transparent;")
        self.text_browser.setGeometry(0, 0, self.parent().width(), self.parent().height())
        self.createButtons()
        self.setVisible(True)
        self.firstPlayer = 0

    def createButtons(self):
        self.choiceButtons = []
        choices = ["🟢", "🔵"]
        for i in range(0,2):
            choiceButton_i = QPushButton(choices[i], self)
            choiceButton_i.setFont(QFont("HP Simplified JPan", 8))
            choiceButton_i.setGeometry(
                (i * 1.355 + 4.25) * self.parent().cellWidth,       # x
                3.15 * self.parent().cellWidth,                     # y
                self.parent().cellWidth/1.65,                       # width
                self.parent().cellWidth/1.65                        # height
            )
            choiceButton_i.setStyleSheet("border: none; background: transparent;")
            choiceButton_i.clicked.connect(functools.partial(self.buttonClicked, i))
            self.choiceButtons.append(choiceButton_i)
            choiceButton_i.show()

    def buttonClicked(self, i):
        self.parent().alphazero._running = True
        self.parent().status = gameStatus.IN_PROGRESS
        if i == 0:
            self.parent().player = CurrentPlayer.HUMAN
        elif i == 1:
            self.parent().player = CurrentPlayer.COMPUTER
            self.parent().alphazero.start()
            self.parent().board.toggleMoveButtons(False)
        for i in range(0,2):
            self.choiceButtons[i].setEnabled(False)
        self.firstPlayer = self.parent().player.value
        self.setVisible(False)
        self.parent().graphicsEffects(False)
        self.parent().timer.start()
        self.parent().timer.timer.start(100)
        self.parent().blurPauseButton(False)

    def drawPieceSelector(self):
        html_content = """
            <html>
            <body style="text-align: center; vertical-align: center;">
            <br>
            <br>
            <br>
            <br>
            <div style="filter: blur(); font-family: HP Simplified JPan">
                <p><span style="font-size: 72px; color: #ffffff;">Choose Your Piece</span></p>
                <p><span style="font-size: 36px; color: #ffffff;">(GREEN goes first)</span></p>
            </div>
            <div style="font-size: 66px;">🟢 🔵</div>
            </body>
            </html>
        """
        self.text_browser.setHtml(html_content)
        self.text_browser.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        self.drawPieceSelector()
        qp.fillRect(self.rect(), QBrush(QColor(0, 0, 0, 100)))
        qp.end()

class pauseGame(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.pauseFlag = False
        self.pauseButton = QPushButton("Pause", self)
        font = QFont("HP Simplified JPan", 16)
        self.pauseButton.setFont(font)
        self.pauseButton.setGeometry(QRect(
            3*self.parent().cellWidth*1/5,
            self.parent().cellWidth*1/6,
            3*self.parent().cellWidth*3/5,
            self.parent().cellWidth*1/2
        ))
        self.pauseButton.clicked.connect(self.pause)
        
    def pause(self):
        self.parent().pauseToggleSignal.emit(self.pauseFlag)
        if not self.pauseFlag:
            self.parent().graphicsEffects(True)
            self.pauseButton.setText("Resume")
            self.parent().timer.timer.stop()
            self.parent().undoMoveButton.undoMoveButton.setEnabled(False)
            self.parent().restartGameButton.restartGameButton.setEnabled(False)
        else:
            self.parent().graphicsEffects(False)
            self.pauseButton.setText("Pause")
            if self.parent().status == gameStatus.IN_PROGRESS:
                self.parent().timer.timer.start()
            self.parent().undoMoveButton.undoMoveButton.setEnabled(True)
            self.parent().restartGameButton.restartGameButton.setEnabled(True)
        self.pauseFlag = not self.pauseFlag

    def paintEvent(self, event):
        qp = QPainter(self)
        if not qp.isActive():
            qp.begin(self)
        qp.fillRect(self.rect(), QBrush(QColor(0, 0, 0)))
        qp.end()

class restartGameButton(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.restartGameButton = QPushButton("Restart", self)
        font = QFont("HP Simplified JPan", 16)
        self.restartGameButton.setFont(font)
        self.restartGameButton.clicked.connect(functools.partial(self.parent().alterGameState.restartAll))
        self.restartGameButton.setGeometry(QRect(
            3*self.parent().cellWidth*1/5,
            self.parent().cellWidth*1/6,
            3*self.parent().cellWidth*3/5,
            self.parent().cellWidth*1/2
        ))

    def paintEvent(self, event):
        qp = QPainter(self)
        if not qp.isActive():
            qp.begin(self)
        qp.fillRect(self.rect(), QBrush(QColor(0, 0, 0)))
        qp.end()

class alterGameState():
    def __init__(self, super):
        self.super = super

    def restartAll(self):
        self.super.status = gameStatus.NOT_STARTED
        self.super.player = CurrentPlayer.NONE
        self.super.alphazero._running = False
        self.super.alphazero.wait()
        self.super.board.state = np.zeros([6,7])
        self.super.board.moveSequence = np.zeros([6,7])
        self.super.board.moveCount = 0
        self.super.timer.resetTimer()
        self.super.board.button_i_status = [True for _ in range(7)]
        self.super.board.toggleMoveButtons(True)
        self.super.graphicsEffects(True)
        self.super.blurPauseButton(True)
        self.super.pieceSelector.setVisible(True)
        for i in range(0,2):
            self.super.pieceSelector.choiceButtons[i].setEnabled(True)

    def undoMove(self):
        if self.super.alphazero.isRunning():
            self.super.alphazero._running = False
            self.super.alphazero.wait()
        last_human_move = np.where(self.super.board.moveSequence[:,:] == np.max(self.super.board.moveSequence))
        last_computer_move = np.where(self.super.board.moveSequence[:,:] == np.min(self.super.board.moveSequence))
        if self.super.board.moveSequence[last_human_move[0][0], last_human_move[1][0]] < abs(self.super.board.moveSequence[last_computer_move[0][0], last_computer_move[1][0]]):
            self.super.board.state[last_computer_move[0][0], last_computer_move[1][0]] = 0
            self.super.board.moveCount -= 1
            self.super.board.moveSequence[last_computer_move[0][0], last_computer_move[1][0]] = 0
        self.super.board.state[last_human_move[0][0], last_human_move[1][0]] = 0
        self.super.board.moveCount -= 1
        self.super.board.moveSequence[last_human_move[0][0], last_human_move[1][0]] = 0
        self.super.board.button_i_status = [True for _ in range(7)]
        for i in list(np.where(self.super.board.state[0, :] != 0)[0]):
            self.super.board.button_i_status[i] = False
        self.super.player = CurrentPlayer.HUMAN
        if self.super.status != gameStatus.IN_PROGRESS:
            self.super.board.moveCount += 1
            self.super.status = gameStatus.IN_PROGRESS
            self.super.timer.timer.start(100)
        self.super.timer.resetTimer()
        self.super.board.update()
        self.super.gameInfo.update()
        self.super.undoMoveButton.update()
        self.super.board.toggleMoveButtons(True)

class undoMoveButton(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.undoMoveButton = QPushButton("Undo", self)
        font = QFont("HP Simplified JPan", 16)
        self.undoMoveButton.setFont(font)
        rect = QRect(
            3*self.parent().cellWidth*1/5,
            self.parent().cellWidth*1/6,
            3*self.parent().cellWidth*3/5,
            self.parent().cellWidth*1/2
        )
        self.undoMoveButton.setGeometry(rect)
        self.undoMoveButton.clicked.connect(self.parent().alterGameState.undoMove)

    def paintEvent(self, event):
        qp = QPainter(self)
        if not qp.isActive():
            qp.begin(self)
        qp.fillRect(self.rect(), QBrush(QColor(0, 0, 0)))
        if self.parent().board.moveCount <= 2:
            self.undoMoveButton.setEnabled(False)
        qp.end()

class mainWindow(QMainWindow):
    gameOverSignal = Signal()
    computerMoveSignal = Signal(dict)
    timeElapsedSignal = Signal(int)
    pauseToggleSignal = Signal(bool)

    def __init__(self):
        super().__init__()
        self.status = gameStatus.NOT_STARTED
        self.cellWidth = 80
        self.playerAction = 0
        self.player = CurrentPlayer.NONE
        self.alterGameState = alterGameState(self)
        random.seed(time.time_ns())
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Connect-Four")
        self.setGeometry(
            2.5 * self.cellWidth,
            0.5 * self.cellWidth,
            10.5 * self.cellWidth,
            7.5 * self.cellWidth
        )
        self.setFixedSize(self.size())

        self.alphazero = AlphaZero(self)
        self.board = Connect4Board(self, self.cellWidth)
        self.timer = Timer(self)
        self.timerGUI = TimerWidget(self, self.cellWidth)
        self.gameInfo = gameInfoWidget(self)
        self.restartGameButton = restartGameButton(self)
        self.pauseGame = pauseGame(self)
        self.pieceSelector = pieceSelector(self)
        self.undoMoveButton = undoMoveButton(self)

        self.computerMoveSignal.connect(self.board.computerMove)
        self.timeElapsedSignal.connect(self.timerGUI.timeElapsed)
        self.gameOverSignal.connect(self.timer.resetTimer)
        self.gameOverSignal.connect(self.board.toggleMoveButtons)
        self.gameOverSignal.connect(self.gameInfo.update)
        self.pauseToggleSignal.connect(self.board.toggleMoveButtons)

        self.pieceSelector.setGeometry(
            0,
            0,
            10.5 * self.cellWidth,
            7.5 * self.cellWidth
        )
        self.board.setGeometry(
            0,
            0,
            7.5 * self.cellWidth,
            7.5 * self.cellWidth
        )
        self.timerGUI.setGeometry(
            7.5 * self.cellWidth,
            0,
            3 * self.cellWidth,
            3 * self.cellWidth
        )
        self.gameInfo.setGeometry(
            7.5 * self.cellWidth,
            3 * self.cellWidth,
            3 * self.cellWidth,
            2 * self.cellWidth
        )
        self.undoMoveButton.setGeometry(
            7.5*self.cellWidth,
            5*self.cellWidth,
            3*self.cellWidth,
            (2.5/3)*self.cellWidth
        )
        self.pauseGame.setGeometry(
            7.5*self.cellWidth,
            (5+(2.5/3))*self.cellWidth,
            3*self.cellWidth,
            (2.5/3)*self.cellWidth
        )
        self.restartGameButton.setGeometry(
            7.5*self.cellWidth,
            (5+5/3)*self.cellWidth,
            3*self.cellWidth,
            (2.5/3)*self.cellWidth
        )

        self.graphicsEffects(True)

        self.board.show()
        self.timerGUI.show()
        self.gameInfo.show()
        self.pieceSelector.show()
        self.restartGameButton.show()
        self.blurPauseButton(True)
        self.pauseGame.show()
        self.undoMoveButton.show()
        self.show()

    def blurPauseButton(self, on):
        if on:
            blurEffect = QGraphicsBlurEffect(self)
            blurEffect.setBlurRadius(10)
            self.pauseGame.setGraphicsEffect(blurEffect)
        else:
            self.pauseGame.setGraphicsEffect(None)

    def graphicsEffects(self, on):
        if on:
            blurEffect = QGraphicsBlurEffect(self)
            blurEffect.setBlurRadius(10)
            self.board.setGraphicsEffect(blurEffect)

            blurEffect = QGraphicsBlurEffect(self)
            blurEffect.setBlurRadius(10)
            self.timerGUI.setGraphicsEffect(blurEffect)

            blurEffect = QGraphicsBlurEffect(self)
            blurEffect.setBlurRadius(10)
            self.gameInfo.setGraphicsEffect(blurEffect)

            blurEffect = QGraphicsBlurEffect(self)
            blurEffect.setBlurRadius(10)
            self.restartGameButton.setGraphicsEffect(blurEffect)

            blurEffect = QGraphicsBlurEffect(self)
            blurEffect.setBlurRadius(10)
            self.undoMoveButton.setGraphicsEffect(blurEffect)

        else:
            self.board.setGraphicsEffect(None)
            self.timerGUI.setGraphicsEffect(None)
            self.gameInfo.setGraphicsEffect(None)
            self.restartGameButton.setGraphicsEffect(None)
            self.undoMoveButton.setGraphicsEffect(None)

    def paintEvent(self, event):
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        qp.fillRect(self.rect(), QBrush(QColor(0, 0, 0)))
        qp.end()

    def switchPlayer(self):
        self.player = CurrentPlayer(0 - self.player.value)