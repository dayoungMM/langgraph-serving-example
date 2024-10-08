# langgraph-serving-example

## 1. 준비
install
- 패키지 설치
```
pip install -r requirements.txt
```

environment variables
- 복사해서 값 넣기
```
cp .env-example .env
```

## 2. LangGraph 개발 가이드
- 추상 클래스 (flows/base.py)
    - LangGraph의 구성요소인 State와 Node를 생성하는 클래스의 추상 클래스 : BaseNode, BaseState
    - 추상 클래스를 상속 가능한 형식으로 개발하면 좋을 것 같습니다.
    - BaseAgent는 플랫폼에서 개발중인 Agent의 추상 클래스로, BaseAgent가 runnable을 만들고 BaseNode에 넣어주는 방식으로 개발중입니다.

- 구현 클래스 (flows/nodes.py)
    - 추상클래스를 상속, 실제 사용할 Node와 State를 만드는 클래스 : MessageNode, MessageState
    - MessageState.message에 이력을 저장하는 방식
    - AIX Platform에서는 이런 방식으로 Graph를 생성하도록 개발할 예정입니다.

- 그래프 예시 - Use Case (flows/graph.py)
    - 구현 클래스로 구현한 Use Case Example
    - 일방혁 서비스 개발자가 작성할 코드 Example

- 실행
    - Visual Studio Code에서 `Run and Debug`에서 🔗run graph 를 선택

## 3. 플랫폼 Agent 사용 가이드
> 플랫폼에서 제공하는 Agent는 App으로 배포한 후 API를 통해 사용할 수 있게 개발중입니다. API Agent 단순 실행 뿐만 아니라 Chaining 해서 사용할 수 있게 할 예정입니다.

- 서버 (app/server.py)  
    - 플랫폼에서 App은 다음과 같이 실행될 예정입니다.
    - Visual Studio Code에서 `Run and Debug`에서 [🍃]run api-server 를 선택해서 실행하세요

- 클라이언트 (app/clent.py)
    - App 사용 방법 예시
    - 실행방법 (서버가 실행되고 있는 상태에서 Project 최상단에서 Terminal로 실행.)
    ```
    python app/client.py
    ```