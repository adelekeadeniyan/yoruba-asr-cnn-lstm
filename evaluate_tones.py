import torch
import re

def analyze_tone_errors(predictions, references):
    """
    Evaluates character and word errors specifically for words containing 
    Yoruba tone marks (acute, grave, macrons, and dot-below diacritics).
    """
    tone_pattern = re.compile(r"[éèíìóòúùẹ́ẹ̀ọ́ọ̀ẹọṣ]")
    
    total_tonal_words = 0
    tonal_errors = 0
    
    for pred, ref in zip(predictions, references):
        ref_words = ref.split()
        pred_words = pred.split()
        
        for r_word in ref_words:
            if tone_pattern.search(r_word):
                total_tonal_words += 1
                if r_word not in pred_words:
                    tonal_errors += 1
                    
    error_rate = tonal_errors / max(1, total_tonal_words)
    print("=== Tonal Error Breakdown ===")
    print(f"Total Tonal Words Evaluated: {total_tonal_words}")
    print(f"Tonal Word Error Rate: {error_rate:.4f}")
    return error_rate

if __name__ == "__main__":
    # Example validation arrays
    sample_preds = ["gbogbo ile eko giga lo ni orisi eka eko"]
    sample_refs = ["gbogbo ilé-ẹ̀kọ̀ gíga ló ní oríṣi ẹ̀ka ẹ̀kọ́."]
    analyze_tone_errors(sample_preds, sample_refs)
