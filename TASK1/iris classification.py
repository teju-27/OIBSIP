import matplotlib.pyplot as plt
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split  
from sklearn.tree import DecisionTreeClassifier, plot_tree  
from sklearn.metrics import accuracy_score
iris_data = load_iris()  
flower_measurements = iris_data.data  
flower_species = iris_data.target  
train_data, test_data, train_labels, test_labels = train_test_split(
    flower_measurements, flower_species, test_size=0.2, random_state=42
)
iris_classifier = DecisionTreeClassifier(random_state=42)
iris_classifier.fit(train_data, train_labels)  
predicted_labels = iris_classifier.predict(test_data)
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"🌿 Model Accuracy: {accuracy:.2f} (out of 1.00)")
new_flower = [[5.1, 3.5, 1.4, 0.2]]  
predicted_species = iris_classifier.predict(new_flower)
predicted_species_name = iris_data.target_names[predicted_species[0]]
print(f"🌸 Predicted Flower Species: {predicted_species_name}")
species_names = iris_data.target_names
actual_counts = [list(test_labels).count(i) for i in range(len(species_names))]
predicted_counts = [list(predicted_labels).count(i) for i in range(len(species_names))]
x = range(len(species_names))
plt.bar(x, actual_counts, width=0.4, label='Actual Count', color='blue', alpha=0.7)
plt.bar([i + 0.4 for i in x], predicted_counts, width=0.4, label='Predicted Count', color='orange', alpha=0.7)
plt.xticks([i + 0.2 for i in x], species_names)
plt.ylabel("Count")
plt.title("Actual vs Predicted Flower Species")
plt.legend()
plt.show()