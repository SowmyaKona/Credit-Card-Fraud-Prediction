import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class CreditCardFraudDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Credit Card Fraud Detection")
        
        # Initialize components
        self.filepath = ""
        
        self.load_button = tk.Button(root, text="Load Dataset", command=self.load_dataset)
        self.load_button.pack(pady=10)

        self.train_button = tk.Button(root, text="Train Model", command=self.train_model, state=tk.DISABLED)
        self.train_button.pack(pady=10)

        self.predict_button = tk.Button(root, text="Predict and Evaluate", command=self.predict_and_evaluate, state=tk.DISABLED)
        self.predict_button.pack(pady=10)
        
        self.result_label = tk.Label(root, text="Accuracy will be displayed here", justify="left")
        self.result_label.pack(pady=20)
        
    def load_dataset(self):
        self.filepath = filedialog.askopenfilename(title="Select Dataset", filetypes=(("CSV Files", "*.csv"),))
        
        if self.filepath:
            try:
                # Load the dataset
                self.credit_card_data = pd.read_csv(self.filepath)
                self.credit_card_data_info()
                self.train_button.config(state=tk.NORMAL)
                self.result_label.config(text="Dataset loaded successfully!\nClick 'Train Model' to proceed.")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading dataset: {e}")
        else:
            messagebox.showinfo("No File", "No file selected!")

    def credit_card_data_info(self):
        # Show basic info about the dataset
        dataset_info = f"Dataset Info:\n"
        dataset_info += f"Shape: {self.credit_card_data.shape}\n"
        dataset_info += f"Columns: {', '.join(self.credit_card_data.columns)}"
        print(dataset_info)
        self.result_label.config(text=dataset_info)

    def train_model(self):
        try:
            # Preprocessing the dataset
            self.credit_card_data['Class'].value_counts()
            legit = self.credit_card_data[self.credit_card_data.Class == 0]
            fraud = self.credit_card_data[self.credit_card_data.Class == 1]

            legit_sample = legit.sample(n=492)
            new_dataset = pd.concat([legit_sample, fraud], axis=0)

            X = new_dataset.drop(columns='Class', axis=1)
            Y = new_dataset['Class']
            
            # Splitting the data
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
            
            # Train the Logistic Regression model
            self.model = LogisticRegression()
            self.model.fit(X_train, Y_train)
            
            self.X_train = X_train
            self.Y_train = Y_train
            self.X_test = X_test
            self.Y_test = Y_test
            
            self.predict_button.config(state=tk.NORMAL)
            self.result_label.config(text="Model trained successfully!\nClick 'Predict and Evaluate' to evaluate the model.")
        except Exception as e:
            messagebox.showerror("Error", f"Error during model training: {e}")
        
    def predict_and_evaluate(self):
        try:
            # Predicting and evaluating the model
            X_train_prediction = self.model.predict(self.X_train)
            training_data_accuracy = accuracy_score(X_train_prediction, self.Y_train)

            X_test_prediction = self.model.predict(self.X_test)
            test_data_accuracy = accuracy_score(X_test_prediction, self.Y_test)

            self.result_label.config(text=f"Training Accuracy: {training_data_accuracy * 100:.2f}%\nTest Accuracy: {test_data_accuracy * 100:.2f}%")
        except Exception as e:
            messagebox.showerror("Error", f"Error during prediction/evaluation: {e}")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = CreditCardFraudDetectionApp(root)
    root.mainloop()
