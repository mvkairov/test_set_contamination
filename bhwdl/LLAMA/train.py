import torch
from model import make_seq_mask
from tqdm.notebook import tqdm
from itertools import repeat


@torch.no_grad()
def generate(model, tokenizer, pad_idx, batch_size=1, prefix=None, max_len=384):
    model.eval()
    if prefix is None:
        prefix = torch.full((batch_size, 1), fill_value=tokenizer.bos_id()).to(next(model.parameters()).device)
    for _ in range(max_len):
        prefix = prefix.clone().detach()
        mask, pad_mask = make_seq_mask(prefix, pad_idx)
        output_logits = torch.nn.functional.softmax(model.forward(prefix, mask, pad_mask)[:, -1, :], dim=-1)
        prefix = torch.cat((prefix, torch.multinomial(output_logits, 1)), dim=-1)
    return prefix


def inf_loop(data_loader):
    for loader in repeat(data_loader):
        yield from loader


def evaluate(model, dataloader, loss_fn, pad_idx):
    model.eval()
    losses = 0
    for tgt, _ in tqdm(dataloader, total=len(dataloader)):
        tgt = tgt.to('cuda')
        tgt_input = tgt[:, :-1]
        mask, pad_mask = make_seq_mask(tgt_input, pad_idx)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = model(tgt_input, mask, pad_mask)
        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(dataloader)


def train(model, n_epochs, pad_idx, optimizer, scheduler, train_loader, val_loader, dataset, wandb_instance, epoch_iter, log_freq=10):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    min_loss = 100

    for epoch in range(n_epochs):
        losses = 0
        cur_step = 0
        for i, (tgt, _) in enumerate(tqdm(inf_loop(train_loader), total=epoch_iter)):
            model.train()
            tgt = tgt.to('cuda')
            tgt_input = tgt[:, :-1]
            mask, padding_mask = make_seq_mask(tgt_input, pad_idx)
            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(tgt_input, mask, padding_mask)
                tgt_out = tgt[:, 1:]
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses += loss.item()
            cur_step += 1

            if i % (epoch_iter // log_freq) == 0 and i > 0:
                wandb_instance.log({
                    'train_loss': losses / i,
                    'lr': scheduler.get_last_lr()[0]
                }, step=cur_step)
                print(f"epoch: {epoch}, train_loss: {(losses / i):.3f}")
                losses = 0

            if i == epoch_iter:
                val_loss = evaluate(model, val_loader, loss_fn, pad_idx)
                if val_loss < min_loss:
                    print(f'checkpoint at {cur_step}')
                    torch.save(model.state_dict(), 'best_model.pt')
                    min_loss = val_loss
                text = dataset.ids2text(generate(model, dataset.sp_model, pad_idx))
                wandb_instance.log({f'stepN{cur_step}': text}, step=cur_step)
                wandb_instance.log({
                    'train_loss': losses / epoch_iter,
                    'val_loss': val_loss,
                    'lr': scheduler.get_last_lr()[0]
                }, step=cur_step)
                print(f'epoch: {epoch}, train_loss: {(losses / epoch_iter):.3f}, val_loss: {val_loss:.3f}')
                break
