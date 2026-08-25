CREATE DATABASE ai_customer_service
    DEFAULT CHARACTER SET utf8mb4;

USE ai_customer_service;

CREATE TABLE orders (
    id INT PRIMARY KEY AUTO_INCREMENT,
    order_no VARCHAR(50) NOT NULL UNIQUE,
    customer_id VARCHAR(50) NOT NULL,
    product_name VARCHAR(200) NOT NULL,
    status VARCHAR(50) NOT NULL,
    tracking_no VARCHAR(100),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO orders
(order_no, customer_id, product_name, status, tracking_no)
VALUES
('10001', 'C001', 'iPhone 17 Pro', '已发货', 'SF123456'),
('10002', 'C002', 'MacBook Pro', '备货中', NULL),
('10003', 'C001', 'AirPods Pro', '已签收', 'SF888888');