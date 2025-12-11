from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


class SupervisedAlgorithms:
    def __init__(self):
        self.models = {}
        self.results = {}

    def train_knn(self, X, y, n_neighbors=5, weights='uniform', algorithm='auto'):
        """Train KNN classifier"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                algorithm=algorithm
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            result = {
                'error': False,
                'algorithm': 'KNN',
                'model': model,
                'predictions': y_pred,
                'test_labels': y_test,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'n_neighbors': n_neighbors,
                'weights': weights,
                'algorithm_param': algorithm,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }

            self.models['KNN'] = model
            self.results['KNN'] = result
            return result

        except Exception as e:
            return {'error': True, 'message': str(e)}

    def train_naive_bayes(self, X, y, var_smoothing=1e-9):
        """Train Naive Bayes classifier"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            model = GaussianNB(var_smoothing=var_smoothing)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            result = {
                'error': False,
                'algorithm': 'Naive Bayes',
                'model': model,
                'predictions': y_pred,
                'test_labels': y_test,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'var_smoothing': var_smoothing,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }

            self.models['Naive Bayes'] = model
            self.results['Naive Bayes'] = result
            return result

        except Exception as e:
            return {'error': True, 'message': str(e)}

    def train_c45(self, X, y, criterion='entropy', max_depth=None, min_samples_split=2):
        """Train C4.5 (Decision Tree) classifier"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # C4.5 uses information gain (entropy)
            model = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            result = {
                'error': False,
                'algorithm': 'C4.5',
                'model': model,
                'predictions': y_pred,
                'test_labels': y_test,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'criterion': criterion,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'feature_importance': model.feature_importances_,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }

            self.models['C4.5'] = model
            self.results['C4.5'] = result
            return result

        except Exception as e:
            return {'error': True, 'message': str(e)}

    def get_model(self, algorithm_name):
        """Get trained model"""
        return self.models.get(algorithm_name)

    def get_result(self, algorithm_name):
        """Get algorithm result"""
        return self.results.get(algorithm_name)

    def get_all_results(self):
        """Get all algorithm results"""
        return self.results

    def clear_results(self):
        """Clear all models and results"""
        self.models.clear()
        self.results.clear()