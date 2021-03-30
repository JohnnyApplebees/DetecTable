import os
import csv
import camelot
import json

from PyPDF2 import PdfFileWriter, PdfFileReader
from pdf2image import convert_from_path, convert_from_bytes

from detectron2.engine import DefaultPredictor

from config import setup_cfg
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import sys

import boto3
from botocore.client import Config
from io import BytesIO, StringIO
import s3fs

from flask import Flask, request, jsonify

from typing import Tuple, Union
from deskew import determine_skew

import cv2
import math

app = Flask(__name__)

s3 = boto3.resource(
		's3',
		# aws_access_key_id=ACCESS_KEY_ID,
		# aws_secret_access_key=ACCESS_SECRET_KEY,
		config=Config(signature_version='s3v4')
	)
@app.route('/detect', methods=['POST'])
def detect():
	main(args, '')
	data = {
		"status":True
	}
	return jsonify(data)


def deskew(image):
	def rotate(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
		old_width, old_height = image.shape[:2]
		angle_radian = math.radians(angle)
		width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
		height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

		image_center = tuple(np.array(image.shape[1::-1]) / 2)
		rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
		rot_mat[1, 2] += (width - old_width) / 2
		rot_mat[0, 2] += (height - old_height) / 2
		return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

	grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	angle = determine_skew(grayscale)
	return angle, rotate(image, angle, (0,0,0))


def get_pdf_info(pdf_file):
	pdf_bytes = pdf_file.get()['Body'].read()
	pdf_doc = PdfFileReader(BytesIO(pdf_bytes))
	pdf_pages = pdf_doc.numPages
	return pdf_bytes, pdf_doc, pdf_pages


def norm_pdf_page(pdf_doc, pg):
	pdf_page = pdf_doc.getPage(pg-1)
	pdf_page.cropBox.upperLeft = (0, list(pdf_page.mediaBox)[-1])
	pdf_page.cropBox.lowerRight = (list(pdf_page.mediaBox)[-2], 0)
	return pdf_page

def pdf_page2img(pdf_file, pg):
	img_page = convert_from_bytes(pdf_file, first_page=pg, last_page=pg)[0]
	return np.array(img_page)


def img_dim(img, bbox):
	H_img,W_img,_=img.shape
	x1_img, y1_img, x2_img, y2_img,_,_=bbox
	w_table, h_table=x2_img-x1_img, y2_img-y1_img
	return [[x1_img, y1_img, x2_img, y2_img], [w_table, h_table], [H_img,W_img]]

def norm_bbox(img, bbox, x_corr=0.05, y_corr=0.05):
	[[x1_img, y1_img, x2_img, y2_img], [w_table, h_table], [H_img,W_img]]=img_dim(img, bbox)
	x1_img_norm,y1_img_norm,x2_img_norm,y2_img_norm=x1_img/W_img, y1_img/H_img, x2_img/W_img, y2_img/H_img
	w_img_norm, h_img_norm=w_table/W_img, h_table/H_img
	w_corr=w_img_norm*x_corr
	h_corr=h_img_norm*x_corr

	return [x1_img_norm-w_corr,y1_img_norm-h_corr/2,x2_img_norm+w_corr,y2_img_norm+2*h_corr]


def bboxes_pdf(img, pdf_page, bbox):
	W_pdf=float(pdf_page.cropBox.getLowerRight()[0])
	H_pdf=float(pdf_page.cropBox.getUpperLeft()[1])

	[x1_img_norm,y1_img_norm,x2_img_norm,y2_img_norm]=norm_bbox(img, bbox)
	x1, y1 = x1_img_norm*W_pdf, (1-y1_img_norm)*H_pdf
	x2, y2 = x2_img_norm*W_pdf, (1-y2_img_norm)*H_pdf

	return [x1, y1, x2, y2]



def output_detectron(img, predictor, cfg):
	prediction = predictor(img)
	outputs = prediction["instances"].get_fields()
	pred_boxes = outputs["pred_boxes"].tensor.cpu().numpy()
	pred_scores = outputs["scores"].cpu().numpy()

	pred_classes = outputs["pred_classes"].cpu().numpy()

	bboxes = []

	classes = {
		0: "Table",
		1: "Table Header" 
	}

	if pred_scores.size:
		for i in range(pred_scores.size):
			bboxes.append([pred_boxes[i][0], pred_boxes[i][1], pred_boxes[i][2], pred_boxes[i][3], classes[pred_classes[i]], pred_scores[i]])

	return bboxes


def parse_args():
	import argparse

	# Parse command line arguments
	ap = argparse.ArgumentParser(description="tables detection")

	ap.add_argument("--config-file",
					required=False,
					default="configs/faster_rcnn_R_101_FPN_3x_client.yaml",
					help="path to config file")

	ap.add_argument("--preprocess",
					action="store_true",
					help="flag to turn on preprocessing")

	ap.add_argument("--confidence-threshold", type=float, default=0.9,
					help="minimum score for instance predictions to be shown (default: 0.9)")
	
	ap.add_argument("opts", default=[], nargs=argparse.REMAINDER,
					help="modify model config options using the command-line")

	return ap.parse_args()


def main(args, event):
	# ACCESS_KEY_ID = 'AKIAXNGPLG4HNVEQXTCY'
	# ACCESS_SECRET_KEY = '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'
	BUCKET_NAME = os.getenv("INPUT_BUCKET", 'test-bucket-2313') or 'test-pdf-bucket1'
	print("BUCKET_NAME", BUCKET_NAME)
	# CSV_BUCKET = os.getenv("OUTPUT_BUCKET") or "test-pdf-bucket1"

	

	my_bucket = s3.Bucket(BUCKET_NAME)
	# csv_bucket = s3.Bucket(CSV_BUCKET)




	if args.confidence_threshold is not None:
		# Set score_threshold for builtin models
		args.opts.append('MODEL.WEIGHTS')
		args.opts.append('model/model_0015999.pth')
		args.opts.append('MODEL.ROI_HEADS.SCORE_THRESH_TEST')
		args.opts.append(str(args.confidence_threshold))
		args.opts.append('MODEL.RETINANET.SCORE_THRESH_TEST')
		args.opts.append(str(args.confidence_threshold))

	cfg = setup_cfg(args)
	predictor = DefaultPredictor(cfg)

	if "Records" in event:
		records = json.loads(event['Records'][0]['body'])
		all_pdfs = []
		for record in records:
			all_pdfs.append(s3.ObjectSummary(BUCKET_NAME, record['s3']['object']['key']))
	else:
		all_bucket_objs = my_bucket.objects.all()
		all_pdfs = [pdf for pdf in all_bucket_objs if pdf.key.lower().endswith(".pdf")]
	print("all_pdfs", all_pdfs)
	print(f"Total PDFs in bucket: {len(all_pdfs)}")


	for i in range(len(all_pdfs)):
		pdf_file = all_pdfs[i].key
		pdfName = os.path.splitext(pdf_file)[0]
		print(f"\n\nProcessing PDF [{i+1}/{len(all_pdfs)}]: {pdf_file}")
		folders = list(my_bucket.objects.filter(Prefix='output/'))
		testFlag = False
		for folder in folders:
			test = folder.key.split('/')
			if len(test) == 3:
				if test[1] == pdfName:
					testFlag = True
					print(f"{pdf_file} already processed!")
					break
		if testFlag:
			continue

		s3_buck = s3.Bucket(BUCKET_NAME)
		folder_name = f"output/{pdfName}"
		s3_buck.put_object(Bucket=BUCKET_NAME, Key=(folder_name+'/'))

		pdf_bytes, pdf_doc, pdf_totalpages = get_pdf_info(all_pdfs[i])
		print(f"     Total Pages: {pdf_totalpages}")

		# Saving the PDF file
		pdf_writer = PdfFileWriter()
		for pg in range(0, pdf_totalpages):
			pdf_writer.addPage(pdf_doc.getPage(pg))
		pdf_processing_path = '/tmp/'+pdfName+'.pdf'
		with open(pdf_processing_path, "wb") as out:
			pdf_writer.write(out)


		for pg in range(1, pdf_totalpages+1):
			print(f"     Processing page [{pg}/{pdf_totalpages}]")	

			pdf_page=norm_pdf_page(pdf_doc, pg)
			img = pdf_page2img(pdf_bytes, pg)
			image_file_path = '/tmp/test.png'
			cv2.imwrite(image_file_path, img)
			img = cv2.imread(image_file_path)
			os.remove(image_file_path)
			angle, img = deskew(img)
			print(f"     Page is rotated at an angle: {angle}°")
			outputs = output_detectron(img, predictor, cfg)

			print(f"               Total Detections: {len(outputs)}")


			interesting_areas=[]
			for x in outputs:
				[x1, y1, x2, y2]=bboxes_pdf(img, pdf_page, x)
				bbox_camelot = [
					",".join([str(x1), str(y1), str(x2), str(y2)])
				][0]  # x1,y1,x2,y2 where (x1, y1) -> left-top and (x2, y2) -> right-bottom in PDF coordinate space
				interesting_areas.append(bbox_camelot)



			# try:
			output_camelot = camelot.read_pdf(filepath=pdf_processing_path, pages=str(pg), flavor="stream", table_areas=interesting_areas)
			output_camelot=[x.df for x in output_camelot]
			if len(output_camelot) == 0:
				print(f"               Camelot unable to extract tables information")
				continue

			for j,db in enumerate(output_camelot):
				csv_buffer = StringIO()
				db.to_csv(csv_buffer, index=False)
				s3_resource = boto3.resource('s3')
				#s3.Bucket(CSV_BUCKET).put_object(Key=(pdfName+'/'))
				bytes_to_write = db.to_csv(None).encode()
				# fs = s3fs.S3FileSystem(key=ACCESS_KEY_ID, secret=ACCESS_SECRET_KEY)
				fs = s3fs.S3FileSystem()
				# fs = s3fs.S3FileSystem()
				with fs.open(f's3://{BUCKET_NAME}/output/{pdfName}/Page {pg} - Table {j+1}.csv', 'wb') as f:
					f.write(bytes_to_write)
			

		# except Exception as e:
		# 	print(e)
		# 	print(f"               [Traceback] Camelot unable to extract tables information")

		os.remove(pdf_processing_path)


if __name__ == "__main__":
	args = parse_args()
	app.run(host='0.0.0.0', port=8080)
	
