// src/Sidebar.js

import React from 'react';
import { Navbar, Nav } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';

const Sidebar = () => {
  return (
    <div style={{ display: 'flex' }}>
      <Navbar bg="dark" variant="dark" className="flex-column" style={{ minHeight: '100vh' }}>
        <Navbar.Brand href="#home">My App</Navbar.Brand>
        <Nav className="flex-column">
          <Nav.Link href="#home">1. Home</Nav.Link>
          <Nav.Link href="#image-generation">2. Image Generation</Nav.Link>
          <Nav.Link href="#image-editing">3. Image Editing</Nav.Link>
        </Nav>
      </Navbar>
    </div>
  );
};

export default Sidebar;
