import requests, bs4, os

url_list = ['https://www.primeshop.com/MIDILibrary/midlist2.htm', 'https://www.primeshop.com/MIDILibrary/midlist3.htm']
download_url = 'https://www.primeshop.com/MIDILibrary/'

save_folder = 'data/complete/ragtimesdzd/'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)   

for url in url_list:
    response = requests.get(url)
    parsed = bs4.BeautifulSoup(response.text, 'html.parser')
    references = parsed.find_all('li')

    for ref in references:
        a = ref.find('a')

        title = a['href']
        if title.endswith('.mid'):
            link_url = requests.compat.urljoin(download_url, title)
            filename = save_folder + title.split('/')[1]
            with open(filename, 'wb') as midifile:
                content = requests.get(link_url).content
                if content <= 30000: #if midi file size <= 30ko
                    midifile.write(content)