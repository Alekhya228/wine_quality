from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators=100, random_state=42)
RF_model.fit(x_train, y_train)

# predictions and confusion matrix
y_pred = RF_model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
labels = sorted(np.unique(np.concatenate((y_test, y_pred))))
ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap='Blues')
plt.show()