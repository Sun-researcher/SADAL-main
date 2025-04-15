def compute_average_precision(y_true, y_pred):

    sorted_indices = sorted(range(len(y_pred)), key=lambda i: y_pred[i], reverse=True)

    true_positives = 0
    num_true = sum(y_true)
    precision_sum = 0

    for i, idx in enumerate(sorted_indices):
        if y_true[idx] == 1:
            true_positives += 1
            precision = true_positives / (i + 1)
            precision_sum += precision

    if num_true == 0:
        average_precision = 0
    else:
        average_precision = precision_sum / num_true

    return average_precision

if __name__=='__main__':
    y_true = [0, 0, 1, 1]
    y_pred = [0.9392216298729181, 0.8955994974821806, 0.922351123765111, 0.9859197791665792]

    ap = compute_average_precision(y_true, y_pred)
    print("Average Precision (AP):", ap)




