from django.shortcuts import render
from .predict import predict

def input_text_view(request):
    if request.method == 'POST':
        input_text = request.POST.get('input_text', '').strip()

        if input_text:  # 입력 텍스트가 있는 경우
            try:
                results = predict(input_text)
                # results는 리스트 형태로, 각 항목은 다음 키를 가진 딕셔너리입니다:
                # 'input_sentence', 'similar_sentence', 'distance', 'cosine_similarity', 'tfidf_similarity'
                return render(request, 'mainapp/results.html', {'results': results, 'input_text': input_text})
            except Exception as e:  # predict 함수 실행 중 예외 발생
                error_message = f"오류가 발생했습니다: {str(e)}"
                return render(request, 'mainapp/input.html', {'error': error_message})
        else:  # 입력 텍스트가 없는 경우
            return render(request, 'mainapp/input.html', {'error': '입력된 텍스트가 없습니다.'})
    else:  # GET 요청인 경우
        return render(request, 'mainapp/input.html')
