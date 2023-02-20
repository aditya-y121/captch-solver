from __future__ import annotations

from sklearn import metrics
from tqdm import tqdm

from model import *
from preprocess import test_targets_orig

device = "cpu"
EPOCHS = 1


def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1]:
                continue
            else:
                fin = fin + j
    return fin


def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k - 1
            if k == -1:
                temp.append("§")
            else:
                p = encoder.inverse_transform([k])[0]
                temp.append(p)
        tp = "".join(temp).replace("§", "")
        cap_preds.append(remove_duplicates(tp))
    return cap_preds


def eval_fn(model, data_loader):
    model.eval()
    fin_loss = 0
    fin_preds = []
    tk0 = tqdm(data_loader, total=len(data_loader))
    for data in tk0:
        for key, value in data.items():
            data[key] = value.to(device)
        batch_preds, loss = model(**data)
        fin_loss += loss.item()
        fin_preds.append(batch_preds)
    return fin_preds, fin_loss / len(data_loader)


if __name__ == "__main__":
    model = Model(num_chars=len(l_enc.classes_))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.8,
        patience=5,
        verbose=True,
    )
    for epoch in tqdm(range(EPOCHS), desc="Training"):
        model.train()
        train_loss = 0
        for img, label in tqdm(train_dl, desc="Training Batch"):
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            _, loss = model.forward(images=img, targets=label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        if (epoch) % 1 == 0:
            model.eval()
            eval_loss = 0
            eval_preds = []
            valid_captcha_preds = []

            for img, label in tqdm(valid_dl, desc="Validating Batch"):
                img, label = img.to(device), label.to(device)
                with torch.no_grad():
                    batch_preds, loss = model.forward(images=img, targets=label)
                    eval_loss += loss.item()
                    eval_preds.append(batch_preds)
            for vp in eval_preds:
                current_preds = decode_predictions(vp, l_enc)
                valid_captcha_preds.extend(current_preds)
            eval_loss /= len(valid_dl)
            test_dup_rem = [remove_duplicates(c) for c in test_targets_orig]
            accuracy = metrics.accuracy_score(test_dup_rem, valid_captcha_preds)
            print(
                f"Epoch={epoch}, Train Loss={train_loss}, Test Loss={eval_loss} Accuracy={accuracy}",
            )
            scheduler.step(eval_loss)
