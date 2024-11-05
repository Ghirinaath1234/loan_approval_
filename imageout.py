from doctr.models import ocr_predictor

def paper_output(doc):
    model = ocr_predictor(pretrained=True)
    result = model(doc)
    json_output = result.export()
    return json_output

def normalize_ocr_data(ocr_data):
    normalized_data=[]
    for page in ocr_data['pages']:
        for block in page['blocks']:
            for line in block.get('lines',[]):
                line_text = ' '.join([word['value'] for word in line['words']])
                confidence_scores = [word['confidence'] for word in line['words']]
                avg_confidence = sum(confidence_scores)/len(confidence_scores)
                normalized_data.append(line_text)
    return normalized_data