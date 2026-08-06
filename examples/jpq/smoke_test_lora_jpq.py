import tempfile

import pyterrier as pt
import pyterrier_dr
import torch
from peft import LoraConfig, TaskType
from pyterrier_dr.jpq import JPQTrainer


def main() -> None:
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # 1. 创建 E5 + LoRA
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["query", "value"],
        bias="none",
    )

    e5 = pyterrier_dr.E5(
        device=device,
        peft_config=lora_config,
    )

    trainable_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in e5.model.named_parameters()
        if parameter.requires_grad
    }

    print("Trainable LoRA tensors:", len(trainable_before))
    print(
        "Trainable LoRA parameters:",
        sum(parameter.numel() for parameter in trainable_before.values()),
    )

    assert trainable_before, "No trainable LoRA parameters were found."

    # 2. 打开已经构建好的 E5 文档索引
    index_path = "tests/fixtures/vaswani_e5.flex"
    index = pyterrier_dr.FlexIndex(index_path)

    print("Index:", index_path)
    print("Indexed documents:", len(index))

    assert len(index) == 11429

    # 3. 创建 JPQ Trainer
    # E5-base 输出 768 维，M=4 可以整除 768。
    trainer = JPQTrainer(
        e5,
        index,
        pq_impl="sklearn",
        M=4,
        nbits=4,
    )

    # 4. 根据 Vaswani qrels 创建正负文档对
    dataset = pt.get_dataset("vaswani")

    doc_pairs = (
        pyterrier_dr.jpq.utils.queries_qrels_to_pairsiter(
            dataset.get_topics(),
            pyterrier_dr.jpq.utils.sample_random_negatives(
                dataset.get_qrels(),
                1,
                index.payload(return_dvecs=False)[0].fwd,
            ),
        )
    )

    # 只使用少量训练对，目的是验证流程，不追求检索效果
    doc_pairs = list(doc_pairs)[:32]

    print("Training document pairs:", len(doc_pairs))

    assert doc_pairs, "No training document pairs were generated."

    print("Example pair:", doc_pairs[0])

    # 5. 运行极小规模的真实 JPQ 训练
    with tempfile.TemporaryDirectory(
        prefix="lora_jpq_smoke_"
    ) as checkpoint_dir:
        trainer.fit(
            doc_pairs,
            total_steps=2,
            patience=1,
            pq_sample_size=256,
            batch_size=4,
            valid_every=1000,
            jpq_negs=1,
            lambda_rank=True,
            in_batch=True,
            checkpoint_dir=checkpoint_dir,
        )

    # 6. 检查 JPQ 训练后 LoRA 参数是否真的变化
    changed_parameters = []

    for name, parameter in e5.model.named_parameters():
        if name not in trainable_before:
            continue

        after = parameter.detach().cpu()

        if not torch.equal(trainable_before[name], after):
            maximum_change = (
                after - trainable_before[name]
            ).abs().max().item()

            changed_parameters.append(
                (name, maximum_change)
            )

    print("\n========== JPQ LoRA update check ==========")
    print(
        "LoRA tensors changed during JPQ training:",
        len(changed_parameters),
    )

    for name, maximum_change in changed_parameters[:10]:
        print(f"{name}: max change = {maximum_change:.10f}")

    assert changed_parameters, (
        "JPQ training completed, but no LoRA parameters changed."
    )

    print("\nE5 + LoRA + JPQ smoke test passed.")


if __name__ == "__main__":
    main()