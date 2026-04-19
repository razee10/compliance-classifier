# compliance-classifier
A tiny web app that:
1. Takes a short compliance-adjacent text (company profile, transaction description, KYC extract)
2. Classifies it: `KYC-relevant` / `AML-relevant` / `sanctions-adjacent` / `not a compliance concern` / `ambiguous`
3. Extracts any red flags with supporting evidence quoted from the doc
4. Returns a confidence signal
