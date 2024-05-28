// src/App.js

// import React from 'react';
// import 'bootstrap/dist/css/bootstrap.min.css';
// import './App.css';
// import Homepage from './pages/Hompage';
// import Sidebar from './components/Sidebar';

// function App() {
//     return (
//         <div style={{ display: 'flex' }}>
//             <Sidebar />
//             <div style={{ flexGrow: 1, padding: '20px' }}>
//             <Homepage />
//             </div>
//         </div>
//     );
// }

// export default App;



// import React from 'react';
// import 'bootstrap/dist/css/bootstrap.min.css';
// import './App.css';
// import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';
// import Homepage from './pages/Hompage';
// import MainContent from './pages/MainContent';
// import Footer from './Footer';
// import Editing from './pages/Editing';


// function App() {
//   return (
//     <div>
//       <header className="site-header sticky-top py-1">
//         <nav className="container d-flex flex-column flex-md-row justify-content-between">
//         <Link className="py-2" to="/" aria-label="Home">
//               <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" className="d-block mx-auto" role="img" viewBox="0 0 24 24">
//                 <title>Home</title>
//                 <circle cx="12" cy="12" r="10" />
//                 <path d="M14.31 8l5.74 9.94M9.69 8h11.48M7.38 12l5.74-9.94M9.69 16L3.95 6.06M14.31 16H2.83m13.79-4l-5.74 9.94" />
//               </svg>
//             </Link>
//             <Link className="py-2 d-none d-md-inline-block" to="/image-generation">Image Generation</Link>
//             <Link className="py-2 d-none d-md-inline-block" to="/image-editing">Image Editing</Link>
//             {/* <Link className="py-2 d-none d-md-inline-block" to="/paper">Paper</Link>
//             <Link className="py-2 d-none d-md-inline-block" to="/code">Code</Link>
//             <Link className="py-2 d-none d-md-inline-block" to="/hugging-space">Hugging Space</Link>
//             <Link className="py-2 d-none d-md-inline-block" to="/colab">Colab</Link> */}
//         </nav>
//       </header>
//       <main>
//         <Routes>
//             <Route path="/" exact component={Homepage} />
//             <Route path="/image-generation" component={MainContent} />
//             <Route path="/image-editing" component={Editing} />
//             {/* Add more routes here if needed */}
//         </Routes>
//       </main>
//       <footer className="container py-5">
//         <Footer />
//       </footer>
//     </div>
//   );
// }

// export default App;


// src/App.js

import React from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom'; // BrowserRouter로 수정
import Homepage from './pages/Hompage';
import MainContent from './pages/MainContent';
import Footer from './Footer';
import Editing from './pages/Editing';

function App() {
  return (
    <Router>
      <div>
        <header className="site-header sticky-top py-1">
          <nav className="container d-flex flex-column flex-md-row justify-content-between">
            <Link className="py-2" to="/" aria-label="Home">
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" className="d-block mx-auto" role="img" viewBox="0 0 24 24">
                <title>Home</title>
                <circle cx="12" cy="12" r="10" />
                <path d="M14.31 8l5.74 9.94M9.69 8h11.48M7.38 12l5.74-9.94M9.69 16L3.95 6.06M14.31 16H2.83m13.79-4l-5.74 9.94" />
              </svg>
            </Link>
            <Link className="py-2 d-none d-md-inline-block" to="/image-generation">Image Generation</Link>
            <Link className="py-2 d-none d-md-inline-block" to="/image-editing">Image Editing</Link>
            <Link className="py-2 d-none d-md-inline-block" to="/paper">Paper</Link>
            <Link className="py-2 d-none d-md-inline-block" to="/code">Code</Link>
            <Link className="py-2 d-none d-md-inline-block" to="/hugging-space">Hugging Space</Link>
            <Link className="py-2 d-none d-md-inline-block" to="/colab">Colab</Link>
          </nav>
        </header>
        <main>
          <Routes>
            <Route path="/" element={<MainContent />} />
            <Route path="/image-generation" element={<Homepage />} />
            <Route path="/image-editing" element={<Editing />} />
            {/* Add more routes here if needed */}
          </Routes>
        </main>
        <footer className="container py-5">
          <Footer />
        </footer>
      </div>
    </Router>
  );
}

export default App;
