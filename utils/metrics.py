def dice_score(pred, target, num_classes=5):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    scores = []
    for cls in range(num_classes):
        p = (pred == cls)
        t = (target == cls)
        inter = (p & t).sum()
        union = p.sum() + t.sum()
        d = 2 * inter / (union + 1e-6)
        scores.append(d)
    return scores