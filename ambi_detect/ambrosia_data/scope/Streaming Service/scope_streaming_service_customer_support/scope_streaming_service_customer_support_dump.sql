BEGIN TRANSACTION;
CREATE TABLE customer_support (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    support_type TEXT,
    description TEXT
);
INSERT INTO "customer_support" VALUES(1,'Live Chat','Real-time chat assistance');
INSERT INTO "customer_support" VALUES(2,'Email Support','Response within 24 hours');
INSERT INTO "customer_support" VALUES(3,'Phone Support','Call us at our toll-free number');
INSERT INTO "customer_support" VALUES(4,'Forum Support','Ask questions and get help from community members');
INSERT INTO "customer_support" VALUES(5,'FAQs','Find answers to common questions');
CREATE TABLE streaming_service (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    price REAL
);
INSERT INTO "streaming_service" VALUES(1,'Netflix',9.99);
INSERT INTO "streaming_service" VALUES(2,'Amazon Prime Video',8.99);
INSERT INTO "streaming_service" VALUES(3,'Hulu',6.99);
INSERT INTO "streaming_service" VALUES(4,'Disney+',7.99);
INSERT INTO "streaming_service" VALUES(5,'HBO Max',14.99);
CREATE TABLE streaming_service_customer_support (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    streaming_service_id INTEGER,
    customer_support_id INTEGER,
    FOREIGN KEY(streaming_service_id) REFERENCES streaming_service(id),
    FOREIGN KEY(customer_support_id) REFERENCES customer_support(id)
);
INSERT INTO "streaming_service_customer_support" VALUES(1,1,1);
INSERT INTO "streaming_service_customer_support" VALUES(2,2,1);
INSERT INTO "streaming_service_customer_support" VALUES(3,3,1);
INSERT INTO "streaming_service_customer_support" VALUES(4,4,1);
INSERT INTO "streaming_service_customer_support" VALUES(5,5,1);
INSERT INTO "streaming_service_customer_support" VALUES(6,5,2);
DELETE FROM "sqlite_sequence";
INSERT INTO "sqlite_sequence" VALUES('streaming_service',5);
INSERT INTO "sqlite_sequence" VALUES('customer_support',5);
INSERT INTO "sqlite_sequence" VALUES('streaming_service_customer_support',6);
COMMIT;
