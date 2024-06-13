// App.js

import React from 'react';
import axios from 'axios';

class App extends React.Component {
    handleClick = () => {
        // Axios를 사용하여 GET 요청 보내기
        axios.get('http://localhost:8000/testparam', {
            // 쿼리 파라미터 설정
            params: {
                url: 'yoojeong'
            }
        })
        .then(response => {
            // 요청 성공 시 처리할 내용
            console.log('응답 데이터:', response.data);
            // 여기서 응답 데이터를 활용하여 상태를 업데이트하거나 필요한 작업을 수행할 수 있습니다.
        })
        .catch(error => {
            // 요청 실패 시 처리할 내용
            console.error('에러 발생:', error);
        });
    };

    render() {
        return (
            <div>
                <button onClick={this.handleClick}>버튼 클릭</button>
            </div>
        );
    }
}

export default App;
