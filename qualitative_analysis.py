def evaluate_qualitative_samples():
    """
    Compares reference target sentences with raw model predictions 
    to highlight tonal marker omission and sequence alignment.
    """
    samples = [
        {
            "reference": "Gbogbo ilé-ẹ̀kọ̀ gíga ló ní oríṣi ẹ̀ka ẹ̀kọ́.",
            "prediction": "gbogbo ile eko giga lo ni orisi eka eko"
        }
    ]
    
    print("=== Qualitative Transcription Comparison ===")
    for idx, sample in enumerate(samples):
        print(f"Sample Index: {idx + 1}")
        print(f"Ground Truth: {sample['reference']}")
        print(f"Prediction:   {sample['prediction']}")
        print("-" * 40)

if __name__ == "__main__":
    evaluate_qualitative_samples()
