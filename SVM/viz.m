% Load CSV data
data = csvread('svm_data.csv', 1, 0); % Skip the header row

% Separate features and labels
X = data(:, 1:2); % Features are in the first two columns
y = data(:, 3);   % Labels are in the third column

% Plot data
figure;
hold on;

% Plot class 0
class_0 = y == 0;
scatter(X(class_0, 1), X(class_0, 2), 50, 'r', 'o', 'DisplayName', 'Class 0');

% Plot class 1
class_1 = y == 1;
scatter(X(class_1, 1), X(class_1, 2), 50, 'b', 'x', 'DisplayName', 'Class 1');

% Label and title
xlabel('Feature 1');
ylabel('Feature 2');
title('Scatter Plot of SVM Data');
legend('show');
grid on;
hold off;
