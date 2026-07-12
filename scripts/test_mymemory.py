from deep_translator import MyMemoryTranslator

try:
    res = MyMemoryTranslator(source='ja', target='es', email="mangatranslatorar@proton.me").translate('こんにちは')
    print(res)
except Exception as e:
    print(e)
