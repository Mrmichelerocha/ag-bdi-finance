import pyautogui
import keyboard
import time

def display_screen_dimensions():
    while True:
        mouse_x, mouse_y = pyautogui.position()
        screen_width, screen_height = pyautogui.size()
        
        time.sleep(10)
        print(f"Dimensões da tela: {screen_width}x{screen_height} na posição do mouse ({mouse_x}, {mouse_y})")

        if keyboard.is_pressed('q'):
            break

display_screen_dimensions()
