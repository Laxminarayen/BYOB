# BYOB
Solution for Profit and loss processing -RRD BYOB
Problem Statement: Conversion of free flowing text description of Line items into a standard description
and extraction of a set of corresponding information from an excel document.

Solution description: The given dataset has a number of PDF documents containing the financial documents and
records of respective companies. The content of each and every document is a scanned image
of the particular financial proof i.e. the bill or the financial statement, in physical form is
scanned as images and then loaded to a PDF file. Each distinguishable PDF document contain
the complete financial record of a company.
As the first step, the documents containing the images of financial records are converted as
images i.e. every single image from the PDF document is converted to separate distinguishable
images. Now these PNG image files containing the financial statements are converted into texts. OCR
(Optical Character Recognition) comes into picture for the conversion of images containing
textual financial statements into text format. Once the textual information are extracted, labeling each and every record (row by row) is
done and then this data is fed into the Lstm network and thereby finding the probability score for each row in the 
label.
