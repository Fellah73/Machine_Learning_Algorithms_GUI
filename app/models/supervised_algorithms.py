from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix , classification_report, precision_score, recall_score, f1_score



class SupervisedAlgorithms:
    def __init__(self):
        self.models = {}
        self.results = {}

        
    def apply_knn_algorithm(self, X, y, params):
        """Apply KNN algorithm"""
        try:
            train_ratio = params.get('training perc', 0.8)
            n_neighbors = params.get('n_neighbors', 5)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_ratio, random_state=42
            )

            # Apply KNN
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)

            # Calculate accuracy for different K values
            k_range = range(1, 11)
            accuracies = []
            for k in k_range:
                knn_temp = KNeighborsClassifier(n_neighbors=k)
                knn_temp.fit(X_train, y_train)
                y_pred_temp = knn_temp.predict(X_test)
                accuracies.append(accuracy_score(y_test, y_pred_temp))

            return {
                'algorithm': 'KNN',
                'model': knn,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'y_pred': y_pred,
                'n_neighbors': n_neighbors,
                'train_ratio': train_ratio,
                'accuracy': accuracy_score(y_test, y_pred),
                'k_accuracies': list(zip(k_range, accuracies)),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
        except Exception as e:
            self.show_error(f"Error in KNN: {str(e)}")
            return None
        
        
    def apply_naive_bayes_algorithm(self, X, y, params):
        """Apply Naive Bayes algorithm"""
        try:
            train_ratio = params.get('training perc', 0.8)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_ratio, random_state=42
            )

            # Apply Naive Bayes
            nb = GaussianNB()
            nb.fit(X_train, y_train)
            y_pred = nb.predict(X_test)

            return {
                'algorithm': 'Naive Bayes',
                'model': nb,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'y_pred': y_pred,
                'train_ratio': train_ratio,
                'accuracy': accuracy_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
        except Exception as e:
            self.show_error(f"Error in Naive Bayes: {str(e)}")
            return None
        
        
    def apply_c45_algorithm(self, X, y, params, feature_names):
        """Apply C4.5 (Decision Tree) algorithm"""
        try:
            train_ratio = params.get('training perc', 0.8)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_ratio, random_state=42
            )

            # Apply Decision Tree (C4.5 approximation)
            dt = DecisionTreeClassifier(
                criterion='entropy',  # C4.5 uses entropy
                max_depth=6,
                min_samples_split=2,
                random_state=42
            )
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)

            return {
                'algorithm': 'C4.5',
                'model': dt,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'y_pred': y_pred,
                'train_ratio': train_ratio,
                'accuracy': accuracy_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'feature_names': feature_names
            }
        except Exception as e:
            self.show_error(f"Error in C4.5: {str(e)}")
            return None


    # Each Algo Classfication Metrics Functions
    def knn_classification_metrics(self, X_train, X_test, y_train, y_test, n_neighbors):
        try:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'algorithm': 'KNN',
                'n_neighbors': n_neighbors
            }
        except Exception as e:
            return {'error': str(e)}
        
    def c45_classification_metrics(self, X_train, X_test, y_train, y_test, max_depth, criterion):
        try:
            dt = DecisionTreeClassifier(
                criterion='entropy',  # C4.5 uses information gain (entropy)
                max_depth=max_depth,
                random_state=42
            )
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'algorithm': 'C4.5',
                'max_depth': max_depth
            }
        except Exception as e:
            return {'error': str(e)}
        
    def naive_bayes_classification(self, X_train, X_test, y_train, y_test, var_smoothing):
        try:
            nb = GaussianNB(var_smoothing=var_smoothing)
            nb.fit(X_train, y_train)
            y_pred = nb.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'algorithm': 'Naive Bayes',
                'var_smoothing': var_smoothing
            }
        except Exception as e:
            return {'error': str(e)}