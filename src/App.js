import React, { useState } from "react";
import Header from "./components/Header";
import Intro from "./components/Intro";
import SearchResults from "./components/SearchResults";
import Error from "./components/Error";
import axios from "axios";

import "./App.css"
import HomeInfo from "./components/HomeInfo";

function App() {

  const apiUrl = "https://fathomless-castle-02888.herokuapp.com/";
  const [pageState, setPageState] = useState(0);
  const [searchResults, setSearchResults] = useState([])
  // 0 - home
  // 1 - intro
  // 200 - success getting api data
  // 500 - api web broke down
  // 1000 - timeout

  function sendRequest(data, callback) {
    const [query, searchTitle, searchAbstract, searchSections] = data;
    var normQuery = '?query=' + query;
    if (!searchTitle) normQuery += '&notitle=1';
    if (!searchAbstract) normQuery += '&noabstract=1';
    if (!searchSections) normQuery += '&nosubsections=1';
    normQuery += '&limit=25';
    const headers = {
      'Content-Type': 'application/json; charset=UTF-8'
    }
    axios.post(
      apiUrl + normQuery, {}, {headers: headers, timeout: 8000}
    ).then(
      response => {
        if(!response.data || !Array.isArray(response.data) || response.data.length <= 0){
          throw 'Invalid response data!'
        }
        setSearchResults(response.data);
        setPageState(200);
        callback();
        console.log('Success', response);
      }
    ).catch(
      error => {
        callback();
        console.log('Error====>', error);
        if(error.response && error.response.status !== 408){
          setPageState(500);
        }
        setPageState(1000);
      }
    )
  }

  function handleClickHome(){
    setPageState(0);
  }

  function handleClickIntro(){
    setPageState(1);
  }

  function Body() {
    if (pageState === 0){
      return <HomeInfo />
    } else if (pageState === 1){
      return <Intro />
    } else if (pageState === 200){
      return <SearchResults data={searchResults}/>
    } else {
      return <Error statusCode={pageState}/>
    }
  }

  return (
    <div>
      <Header onSearch={sendRequest} onClickHome={handleClickHome} onClickIntro={handleClickIntro}/>
      <Body/>
    </div>
  );
}

export default App;
