
import streamlit as st
import pandas as pd
import os
import warnings
import re
from io import BytesIO
from typing import List, Dict, Any

# LangChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFium2Loader, Docx2txtLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 변경됨
from langchain_huggingface import HuggingFaceEmbeddings  # 변경됨
from langchain_community.vectorstores import FAISS  # 변경됨
from langchain.chains.retrieval_qa.base import RetrievalQA  # 변경됨

def main():
    st.set_page_config(
        page_title="지리과 서답형 문항 자동채점 플랫폼",
        page_icon=":memo:"
    )

    st.title(":world_map: 지리과 서답형 문항 자동채점 플랫폼")
    
    # 세션 상태 초기화
    if "reference_docs" not in st.session_state:
        st.session_state.reference_docs = None
    if "grading_rubric" not in st.session_state:
        st.session_state.grading_rubric = None
    if "model_ready" not in st.session_state:
        st.session_state.model_ready = False

    # 사이드바 설정
    with st.sidebar:
        st.header("1. 참고 문서 업로드")
        uploaded_files = st.file_uploader(
            "모범 답안, 채점 기준, 참고 문서를 업로드하세요 (PDF, Word, Excel)",
            type=['pdf', 'docx', 'xlsx'],
            accept_multiple_files=True
        )
        
        st.header("2. 모델 설정")
        model_choice = st.radio(
            "사용할 LLM 모델 선택",
            ["Groq", "Gemini"]
        )
        
        api_key = st.text_input(
            f"{model_choice} API 키",
            type="password"
        )
        
        process_btn = st.button("문서 처리하기")
        
        if process_btn and uploaded_files:
            if not api_key:
                st.error("API 키를 입력해주세요.")
                st.stop()
                
            with st.spinner("문서를 처리 중입니다..."):
                try:
                    # 문서 로드 및 처리
                    docs = load_documents(uploaded_files)
                    st.write(f"로드된 문서 수: {len(docs)}")
                    
                    if not docs:
                        st.error("문서를 로드하지 못했습니다.")
                        st.stop()
                    
                    text_chunks = split_documents(docs)
                    st.write(f"분할된 청크 수: {len(text_chunks)}")
                    
                    if not text_chunks:
                        st.error("문서를 분할하지 못했습니다.")
                        st.stop()
                    
                    vectorstore = create_vectorstore(text_chunks)
                    
                    # 벡터 저장소의 상태를 확인
                    if not hasattr(vectorstore, 'index') or vectorstore.index is None:
                        st.error("벡터 저장소가 초기화되지 않았습니다.")
                        st.stop()
                    
                    # 임베딩된 벡터 수 확인
                    try:
                        # FAISS의 저장된 벡터 수 확인
                        num_embeddings = vectorstore.index.ntotal
                        st.write(f"임베딩된 문서 수: {num_embeddings}")
                        
                        # 임베딩된 벡터의 차원 확인
                        if hasattr(vectorstore.index, 'd'):
                            st.write(f"임베딩 차원: {vectorstore.index.d}")
                    except Exception as e:
                        st.error(f"벡터 저장소 상태 확인 중 오류: {str(e)}")
                        st.stop()
                    
                    # 모델 초기화
                    if model_choice == "Groq":
                        # API 키를 환경 변수로 설정
                        os.environ["GROQ_API_KEY"] = api_key
                        llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
                    else:  # Gemini
                        llm = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-pro")
                    
                    # 검색 파라미터를 출력하여 확인
                    search_kwargs = {
                        "k": 10,
                        "score_threshold": 0.3,
                        "search_type": "similarity_score_threshold"
                    }
                    
                    # 벡터 저장소의 상태를 확인
                    if not hasattr(vectorstore, 'index') or vectorstore.index is None:
                        st.error("벡터 저장소가 초기화되지 않았습니다.")
                        st.stop()
                    
                    # 검색기의 상태를 확인
                    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
                    
                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever(
                            search_kwargs={
                                "k": 10,  # 반환할 문서 수
                                "score_threshold": 0.3,  # 유사도 임계값
                                "search_type": "similarity"  # 검색 방식
                            }
                        ),
                        return_source_documents=True
                    )
                    
                    st.session_state.model_ready = True
                    st.success("문서 처리가 완료되었습니다!")
                    
                except Exception as e:
                    st.error(f"문서 처리 중 오류가 발생했습니다: {str(e)}")

    # 메인 화면
    if st.session_state.model_ready:
        st.header("학생 답안 채점")
        
        # 채점 기준 입력
        st.subheader("채점 기준")
        criteria = st.text_area(
            "채점 기준을 입력하세요",
            height=100
        )
        
        # 모범 답안 입력
        st.subheader("모범 답안")
        model_answer = st.text_area(
            "모범 답안을 입력하세요",
            height=150
        )
        
        # 학생 답안 입력
        st.subheader("학생 답안")
        student_answers = []
        
        # 다중 학생 답안 입력
        num_students = st.number_input(
            "학생 수",
            min_value=1,
            max_value=50,
            value=1,
            step=1
        )
        
        for i in range(num_students):
            st.write(f"### 학생 {i+1}")
            name = st.text_input(f"학생 이름 {i+1}", key=f"name_{i}")
            answer = st.text_area(f"답안 {i+1}", key=f"answer_{i}")
            student_answers.append({"name": name, "answer": answer})
        
        # 채점 실행
        if st.button("채점하기"):
            if not criteria or not model_answer:
                st.warning("채점 기준과 모범 답안을 모두 입력해주세요.")
            else:
                results = []
                with st.spinner("채점 중입니다..."):
                    for student in student_answers:
                        if student["answer"]:  # 답변이 있는 경우에만 채점
                            feedback = evaluate_answer(
                                student["answer"],
                                model_answer,
                                criteria,
                                st.session_state.qa_chain
                            )
                            results.append({
                                "이름": student["name"],
                                "답안": student["answer"],
                                "점수": feedback["score"],
                                "피드백": feedback["feedback"],
                                "개선 제안": feedback["suggestions"],
                                "참고 문서": "\n\n---\n\n".join([doc['content'] for doc in feedback.get("sources", [])]) if feedback.get("sources") else "참고 문서 없음"
                            })
        
                # 결과 표시
                if results:
                    st.subheader("채점 결과")
                
                    # 각 학생별로 상세 결과 표시
                    for i, result in enumerate(results):
                        with st.expander(f"{result['이름']} 학생의 상세 결과 보기", expanded=(i==0)):
                            st.write(f"### 점수: {result['점수']}")
                            
                            st.write("#### 답안")
                            st.write(result['답안'])
                            
                            st.write("#### 피드백")
                            st.write(result['피드백'])
                            
                            st.write("#### 개선 제안")
                            st.write(result['개선 제안'])
                    
                            # 참고 문서 표시
                            if result.get('sources'):
                                st.write("---")
                                st.write("#### 참고한 문서")
                                for doc in result['sources']:
                                    with st.expander(f"문서 {doc['source']}"):
                                        st.write(doc['content'])
                                        st.write(f"문서 길이: {doc['length']}자")

            
                    # 전체 결과를 데이터프레임으로 표시
                    st.subheader("전체 결과")
                    
                    # 결과 데이터프레임 생성
                    df = pd.DataFrame(results)
                    
                    # 모든 문자열 컬럼 정제
                    for col in df.select_dtypes(include=['object']).columns:
                        df[col] = df[col].astype(str).apply(clean_text)
                    
                    # 데이터프레임을 엑셀 파일로 변환
                    excel_file = BytesIO()
                    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='채점 결과')
                        
                        # 워크시트 조정
                        worksheet = writer.sheets['채점 결과']
                        for idx, col in enumerate(df.columns):
                            max_length = max(
                                df[col].astype(str).map(len).max(),
                                len(str(col))
                            ) + 2
                            worksheet.column_dimensions[chr(65+idx)].width = min(max_length, 50)
                    
                    processed_data = excel_file.getvalue()
                    
                    # 결과 표시
                    st.dataframe(df)
                    
                    # 엑셀 다운로드 버튼
                    st.download_button(
                        label="결과 다운로드 (Excel)",
                        data=processed_data,
                        file_name="채점_결과.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="excel_download_button"  # 고유한 key 추가
                    )


                    
def load_documents(uploaded_files):
    """업로드된 문서들을 로드합니다."""
    docs = []
    if not uploaded_files:
        st.warning("업로드된 파일이 없습니다.")
        return docs
        
    for file in uploaded_files:
        file_path = f"./temp_{file.name}"
        st.write(f"처리 중인 파일: {file.name}")
        
        try:
            # 임시 파일로 저장
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            
            # 파일 크기 확인 (디버깅용)
            file_size = os.path.getsize(file_path) / 1024  # KB 단위로 변환
            st.write(f"- 파일 크기: {file_size:.2f} KB")
            
            # 파일 확장자에 따른 로더 선택
            loaded_docs = []
            if file.name.lower().endswith('.pdf'):
                st.write("- PDF 문서를 로드하는 중...")
                loader = PyPDFium2Loader(file_path)
                loaded_docs = loader.load()
            elif file.name.lower().endswith('.docx'):
                st.write("- Word 문서를 로드하는 중...")
                loader = Docx2txtLoader(file_path)
                loaded_docs = loader.load()
            elif file.name.lower().endswith('.xlsx'):
                st.write("- Excel 문서를 로드하는 중...")
                loader = UnstructuredExcelLoader(file_path)
                loaded_docs = loader.load()
            else:
                st.warning(f"지원하지 않는 파일 형식입니다: {file.name}")
                continue
                
            # 로드된 문서 정보 출력
            if not loaded_docs:
                st.warning(f"문서에서 내용을 추출하지 못했습니다: {file.name}")
                continue
                
            st.write(f"- 추출된 문서 수: {len(loaded_docs)}")
            for i, doc in enumerate(loaded_docs[:3]):  # 처음 3개 문서만 샘플로 출력
                preview = doc.page_content[:100].replace('\n', ' ').strip()
                st.write(f"  문서 {i+1} 미리보기: {preview}...")
            if len(loaded_docs) > 3:
                st.write(f"  ... 외 {len(loaded_docs)-3}개 문서 생략")
                
            docs.extend(loaded_docs)
            
        except Exception as e:
            st.error(f"문서 로드 중 오류 발생 ({file.name}): {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            
        finally:
            # 임시 파일 정리
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                st.warning(f"임시 파일 삭제 중 오류 발생: {str(e)}")
    
    st.write(f"총 로드된 문서 수: {len(docs)}")
    return docs

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    """문서를 청크로 분할합니다."""
    if not docs:
        st.warning("분할할 문서가 없습니다.")
        return []
    
    st.write("\n## 문서 분할 진행 상황")
    st.write(f"- 청크 크기: {chunk_size}자")
    st.write(f"- 청크 중복 크기: {chunk_overlap}자")
    
    # 텍스트 분할기 초기화
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,  # 청크 시작 인덱스 추가
        separators=['\n\n', '\n', ' ', ''],  # 한국어에 더 적합한 구분자 설정
        keep_separator=True  # 구분자 유지
    )
    
    try:
        # 문서 분할
        chunks = text_splitter.split_documents(docs)
        
        if not chunks:
            st.error("문서를 분할하는 데 실패했습니다.")
            return []
            
        st.write(f"- 원본 문서 수: {len(docs)}")
        st.write(f"- 생성된 청크 수: {len(chunks)}")
        
        # 분할된 청크 샘플 출력 (처음 3개)
        st.write("\n### 분할된 청크 샘플:")
        for i, chunk in enumerate(chunks[:3]):
            preview = chunk.page_content[:150].replace('\n', ' ').strip()
            st.write(f"**청크 {i+1}** (길이: {len(chunk.page_content)}자):")
            st.write(f"> {preview}...")
        
        if len(chunks) > 3:
            st.write(f"... 외 {len(chunks)-3}개 청크 생략")
        
        # 분할 통계 계산
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
        
        st.write("\n### 분할 통계:")
        st.write(f"- 평균 청크 길이: {avg_length:.1f}자")
        st.write(f"- 최소 청크 길이: {min(chunk_lengths) if chunk_lengths else 0}자")
        st.write(f"- 최대 청크 길이: {max(chunk_lengths) if chunk_lengths else 0}자")
        
        # 분할된 청크들에 메타데이터 추가
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['source'] = chunk.metadata.get('source', f"chunk_{i}")
            chunk.metadata['length'] = len(chunk.page_content)
        
        return chunks
        
    except Exception as e:
        st.error(f"문서 분할 중 오류 발생: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return []

def create_vectorstore(docs):
    """벡터 저장소를 생성합니다."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        import torch
        
        st.write("CPU/GPU 확인 중...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.write(f"사용 중인 장치: {device}")
        
        # 더 가벼운 한국어 모델로 변경
        model_name = "jhgan/ko-sroberta-multitask"
        st.write(f"임베딩 모델 로드 중: {model_name}")
        
        try:
            # 간소화된 임베딩 설정
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={
                    "normalize_embeddings": True,
                }
            )
            embeddings.show_progress = False
            st.success("임베딩 모델 로드 완료")
        except Exception as e:
            st.error(f"임베딩 모델 로드 실패: {str(e)}")
            raise
        
        # 문서 내용이 비어있지 않은지 확인
        valid_docs = [doc for doc in docs if doc.page_content.strip()]
        if not valid_docs:
            st.error("유효한 문서 내용이 없습니다.")
            return None
            
        st.write(f"처리할 문서 수: {len(valid_docs)}")
        
        try:
            # 문서를 작은 배치로 나누어 처리
            batch_size = 5
            vectorstore = None
            
            st.write("벡터 저장소 생성 중...")
            for i in range(0, len(valid_docs), batch_size):
                batch = valid_docs[i:i + batch_size]
                st.write(f"처리 중: {i+1}-{min(i+len(batch), len(valid_docs))}/{len(valid_docs)}")
                
                try:
                    if vectorstore is None:
                        vectorstore = FAISS.from_documents(batch, embeddings)
                    else:
                        temp_store = FAISS.from_documents(batch, embeddings)
                        vectorstore.merge_from(temp_store)
                except Exception as e:
                    st.warning(f"일부 문서 처리 중 오류 발생: {str(e)}")
                    continue
            
            if vectorstore is not None and hasattr(vectorstore, 'index'):
                st.success(f"벡터 저장소가 성공적으로 생성되었습니다. (문서 수: {len(valid_docs)})")
                return vectorstore
            else:
                st.error("벡터 저장소 생성에 실패했습니다.")
                return None
                
        except Exception as e:
            st.error(f"벡터 저장소 생성 중 오류: {str(e)}")
            raise
                
    except Exception as e:
        st.error(f"벡터 저장소 생성 중 치명적 오류 발생: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

def evaluate_answer(student_answer, model_answer, criteria, qa_chain):
    """학생 답안을 평가합니다."""
    prompt_template = """
    당신은 지리 교사입니다. 다음 정보를 바탕으로 학생의 답안을 평가해주세요.
    
    [채점 기준]
    {criteria}

    [모범 답안]
    {model_answer}
    
    [학생 답안]
    {student_answer}
    
    [참고 문서]
    {source_docs}
    
    다음 형식으로 답변해주세요:
    점수: "채점 기준"에 따라 평가하여 도출한 점수를 합산해주세요. 이때, 평가 기준에만 해당하면 최대한 점수를 부여하세요.
    피드백: "채점 기준"을 바탕으로 학생의 답안을 평가한 결과를 "참고 문서"를 통해 설명해주세요. "채점 기준"에서 제시하는 영역별로 나누어 피드백을 제공합니다.
    """
    
    # 관련 문서 검색
    try:
        st.write("관련 문서 검색 중...")
        
        # 벡터 저장소에서 유사한 문서 검색
        if hasattr(qa_chain, 'retriever') and hasattr(qa_chain.retriever, 'vectorstore'):
            vectorstore = qa_chain.retriever.vectorstore
            related_docs = vectorstore.similarity_search_with_score(
                student_answer,
                k=3  # 상위 3개 문서만 가져오기
            )
            
            st.write(f"검색된 문서 수: {len(related_docs)}")
            
            # 문서와 점수 분리
            docs_with_scores = []
            for doc, score in related_docs:
                doc.metadata['score'] = float(score)  # 점수를 metadata에 저장
                docs_with_scores.append(doc)
                st.write(f"문서 유사도: {score:.2f}")
                st.write(f"문서 내용: {doc.page_content[:200]}...")
                
            related_docs = docs_with_scores
        else:
            st.warning("벡터 저장소를 찾을 수 없습니다. 문서 기반 검색을 건너뜁니다.")
            related_docs = []
            
    except Exception as e:
        st.error(f"문서 검색 중 오류 발생: {str(e)}")
        related_docs = []
    
    # 문서 내용을 문자열로 변환
    source_docs_text = "\n\n---\n\n".join([doc.page_content for doc in related_docs]) if related_docs else "관련 문서를 찾을 수 없습니다."
    
    try:
        # 프롬프트 생성 및 모델 호출
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["criteria", "model_answer", "student_answer", "source_docs"]
        )
        
        response = qa_chain.invoke({
            "query": prompt.format(
                criteria=criteria,
                model_answer=model_answer,
                student_answer=student_answer,
                source_docs=source_docs_text
            ),
            "return_source_documents": True
        })
        
        # 응답 파싱
        score = 0
        feedback = ""
        suggestions = ""
        
        if isinstance(response, dict):
            response_text = response.get('result', str(response))
        else:
            response_text = str(response)
            
        # 점수 추출
        score_match = re.search(r'점수\s*:\s*(\d+)', response_text)
        if score_match:
            score = int(score_match.group(1))
            
        # 피드백 추출
        feedback_match = re.search(r'피드백\s*:\s*(.+?)(?=\n\S+:|$)', response_text, re.DOTALL)
        if feedback_match:
            feedback = feedback_match.group(1).strip()
            
        # 개선 제안 추출
        suggestion_match = re.search(r'개선\s*제안\s*:\s*(.+?)(?=\n\S+:|$)', response_text, re.DOTALL)
        if suggestion_match:
            suggestions = suggestion_match.group(1).strip()
            
        return {
            "score": score,
            "feedback": feedback or "피드백을 생성할 수 없습니다.",
            "suggestions": suggestions or "개선 제안을 생성할 수 없습니다.",
            "sources": [{"content": doc.page_content, "score": doc.metadata.get('score', 0)} for doc in related_docs]
        }
        
    except Exception as e:
        st.error(f"응답 처리 중 오류 발생: {str(e)}")
        return {
            "score": 0,
            "feedback": "평가 중 오류가 발생했습니다.",
            "suggestions": "잠시 후 다시 시도해주세요.",
            "sources": []
        }
   

def clean_text(text):
    """Excel에 저장할 수 없는 문자를 제거합니다."""
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return ""
    # 제어 문자 및 문제가 될 수 있는 특수 문자 제거
    cleaned = []
    for char in text:
        if 32 <= ord(char) <= 126 or 0xAC00 <= ord(char) <= 0xD7A3:
            cleaned.append(char)
        else:
            cleaned.append(' ')
    return ''.join(cleaned).strip()

def create_excel_download_link(df, filename="채점_결과.xlsx"):
    """데이터프레임을 엑셀 파일로 변환하여 다운로드 링크를 생성합니다."""
    try:
        # 데이터프레임을 엑셀 파일로 변환
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='채점 결과')
        
        # 다운로드 링크 생성
        b64 = base64.b64encode(output.getvalue()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">엑셀 파일로 다운로드</a>'
        return href
    except Exception as e:
        st.error(f"엑셀 파일 생성 중 오류가 발생했습니다: {str(e)}")
        return None

if __name__ == "__main__":
    main()