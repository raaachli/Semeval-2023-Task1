import openai
import csv
import time
openai.api_key = 'xxx' # enter your openai api key here


def get_gpt_text():
	data = 'train'
	data_dir = '/media/kiki/971339f7-b775-448b-b7d8-f17bc1499e4d/Dataset/semeval-2023-test/test.data.v1.1/'
	# save_file_a = data_dir + data + '_v1/' + data + '.gpt.a_1.txt'
	# save_file_b = data_dir + data + '_v1/' + data + '.gpt.b_1.txt'
	# save_file_ab = data_dir + data + '_v1/' + data + '.gpt.ab.txt'

	text_file = data_dir + 'fa.test.data.txt'
	save_file_ab = data_dir + 'fa.en2.gpt.ab.txt'
	save_file_b = data_dir + 'en.gpt.b.txt'
	save_file_e = data_dir + 'en.gpt.e.txt'

	with open(text_file) as f:
		reader = csv.reader(f, delimiter="\t")
		data = list(reader)

	# data = ['personal bag',
	#         'ecstasy feeling',
	#         'metal nail',
	#         'arable field',
	#         'living abode',
	#         'ecstasy drug',
	#         'nail hand']

	resume = 0
	end = len(data)

	for index in range(resume, end):
		print(index)
		# text_a = 'Describe "'+ data[index][0]+'" ' + 'in one sentence: '
		#
		# text_phrase = data[index][1]
		# des_phrase = text_phrase.replace(data[index][0], '')
		# text_b = 'Describe "' + des_phrase +'" ' + 'in one sentence: '
		#
		# prompt_a = text_a
		# prompt_b = text_b

		# text_ab = 'Describe "' + data[index][1]+'" ' + 'in one sentence: '  # english
		# text_ab = 'Descrivere "' + data[index][1]+'" ' + 'in una frase: '  # italian
		# print(data[index][1])
		text_ab = 'Describe "' + data[index][1]+'" ' + 'in one sentence: '  # fra

		prompt_ab = text_ab

		try:
			# time.sleep(2)
			# response = openai.Completion.create(engine='text-davinci-003', prompt = prompt_a, max_tokens = 50)
			# response_text = response['choices'][0]['text']
			# response_text = response_text.strip()
			# with open(save_file_a, 'a') as f:
			# 	f.write(response_text + '\n')

			time.sleep(2)
			response = openai.Completion.create(engine='text-davinci-003', prompt=prompt_ab, max_tokens=50)
			response_text = response['choices'][0]['text']
			response_text = response_text.strip()
			with open(save_file_ab, 'a') as f:
				f.write(response_text + '\n')

		except Exception as e:
			print('exception')
			print(e)

			# time.sleep(5)
			# response = openai.Completion.create(engine='text-davinci-003', prompt=prompt_a, max_tokens=50)
			# response_text = response['choices'][0]['text']
			# response_text = response_text.strip()
			# with open(save_file_a, 'a') as f:
			# 	f.write(response_text + '\n')

			time.sleep(5)
			response = openai.Completion.create(engine='text-davinci-003', prompt=prompt_ab, max_tokens=50)
			response_text = response['choices'][0]['text']
			response_text = response_text.strip()
			with open(save_file_ab, 'a') as f:
				f.write(response_text + '\n')

			pass


if __name__ == '__main__':
	get_gpt_text()
