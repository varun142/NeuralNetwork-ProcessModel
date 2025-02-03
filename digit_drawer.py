import tkinter as tk
import numpy as np
import cv2
from PIL import Image, ImageDraw
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time


class DigitDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")

        self.canvas = tk.Canvas(root, width=200, height=200, bg="black")  # 20x20 grid
        self.canvas.pack()
        
        self.image = Image.new("L", (200, 200), "black")  # PIL image for drawing
        self.draw = ImageDraw.Draw(self.image)
        
        self.canvas.bind("<B1-Motion>", self.paint)  # Bind mouse movement
        self.predict_btn = tk.Button(root, text="Predict", command=self.predict_digit)
        self.predict_btn.pack()
        
        self.reset_btn = tk.Button(root, text="Reset", command=self.reset_canvas)
        self.reset_btn.pack()

        # Create a Matplotlib figure for dynamic updates
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_fig.get_tk_widget().pack()

        self.last_prediction_time = 0  # Track the last prediction time

    def paint(self, event):
        x, y = event.x, event.y
        r = 8  # Brush radius
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill="white", outline="white")
        
        
        current_time = int(time.time() * 1000)  # Get current time in milliseconds
        if current_time - self.last_prediction_time > 100:  # Predict every 100ms
            self.predict_digit()
            self.last_prediction_time = current_time
    
    def reset_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 200, 200], fill="black")
        self.ax.clear()
        self.canvas_fig.draw()

    def preprocess(self):
        # Resize to 28x28 like MNIST
        img = self.image.resize((28, 28))
        img = np.array(img)
        img = img / 255.0  # Normalize
        img = img.reshape(1, 28, 28, 1)  # Reshape for the model
        return img

    def predict_digit(self):
        img = self.preprocess()
        pred = model.predict(img)
        predicted_label = np.argmax(pred)
        print(f"Predicted Digit: {predicted_label}")
        
        # Visualize weights after prediction
        self.visualize_weights()

    def visualize_weights(self):
        # Clear the previous plot
        self.ax.clear()
        
        # Get the weights of the first and second hidden layers
        weights1 = model.get_layer("hidden_1").get_weights()[0]
        weights2 = model.get_layer("hidden_2").get_weights()[0]
        
        # Plot the weights as boxes with lines connecting them
        self.plot_weights(weights1, weights2)
        
        # Draw the updated figure
        self.canvas_fig.draw()

    def plot_weights(self, weights1, weights2):
        # Number of neurons in each layer
        input_neurons = weights1.shape[0]
        hidden1_neurons = weights1.shape[1]
        hidden2_neurons = weights2.shape[1]

        # Plot input layer neurons
        for i in range(input_neurons):
            self.ax.add_patch(plt.Rectangle((0, i), 1, 1, fill=True, color='blue'))

        # Plot hidden layer 1 neurons
        for j in range(hidden1_neurons):
            self.ax.add_patch(plt.Rectangle((2, j), 1, 1, fill=True, color='green'))
            for i in range(input_neurons):
                self.ax.plot([1, 2], [i + 0.5, j + 0.5], 'k-', lw=weights1[i, j])

        # Plot hidden layer 2 neurons
        for k in range(hidden2_neurons):
            self.ax.add_patch(plt.Rectangle((4, k), 1, 1, fill=True, color='red'))
            for j in range(hidden1_neurons):
                self.ax.plot([3, 4], [j + 0.5, k + 0.5], 'k-', lw=weights2[j, k])

        # Set plot limits and labels
        self.ax.set_xlim(-1, 5)
        self.ax.set_ylim(-1, max(input_neurons, hidden1_neurons, hidden2_neurons))
        self.ax.set_aspect('equal')
        self.ax.axis('off')

# Initialize Tkinter
root = tk.Tk()
app = DigitDrawer(root)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu', name="hidden_1"),
    tf.keras.layers.Dense(64, activation='relu', name="hidden_2"),
    tf.keras.layers.Dense(10, activation='softmax', name="output")
])

# Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load and train on MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

print("\nModel Trained Successfully!")

root.mainloop()
