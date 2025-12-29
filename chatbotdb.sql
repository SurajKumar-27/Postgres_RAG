--
-- PostgreSQL database dump
--

-- Dumped from database version 17.5
-- Dumped by pg_dump version 17.5

-- Started on 2025-12-29 17:08:17

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- TOC entry 4 (class 2615 OID 2200)
-- Name: public; Type: SCHEMA; Schema: -; Owner: pg_database_owner
--

CREATE SCHEMA public;


ALTER SCHEMA public OWNER TO pg_database_owner;

--
-- TOC entry 5254 (class 0 OID 0)
-- Dependencies: 4
-- Name: SCHEMA public; Type: COMMENT; Schema: -; Owner: pg_database_owner
--

COMMENT ON SCHEMA public IS 'standard public schema';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- TOC entry 296 (class 1259 OID 29120)
-- Name: chat_messages; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.chat_messages (
    id integer NOT NULL,
    session_id character varying NOT NULL,
    role character varying NOT NULL,
    content text NOT NULL,
    created_at timestamp without time zone
);


ALTER TABLE public.chat_messages OWNER TO postgres;

--
-- TOC entry 295 (class 1259 OID 29119)
-- Name: chat_messages_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.chat_messages_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.chat_messages_id_seq OWNER TO postgres;

--
-- TOC entry 5255 (class 0 OID 0)
-- Dependencies: 295
-- Name: chat_messages_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.chat_messages_id_seq OWNED BY public.chat_messages.id;


--
-- TOC entry 294 (class 1259 OID 29112)
-- Name: chat_sessions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.chat_sessions (
    id character varying NOT NULL,
    created_at timestamp without time zone
);


ALTER TABLE public.chat_sessions OWNER TO postgres;

--
-- TOC entry 217 (class 1259 OID 17872)
-- Name: ezc_buss_partner_config; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_buss_partner_config (
    ebpc_buss_partner character varying(10),
    ebpc_catalog_no numeric(6,0),
    ebpc_multiple_sales_areas character varying(1),
    ebpc_price_selection_flag character varying(1),
    ebpc_unlimited_users character varying(1),
    ebpc_number_of_users numeric(10,0),
    epbc_intranet_flag character varying(1)
);


ALTER TABLE public.ezc_buss_partner_config OWNER TO postgres;

--
-- TOC entry 218 (class 1259 OID 17875)
-- Name: ezc_buss_partner_params; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_buss_partner_params (
    ebpp_buss_partner character varying(10),
    ebpp_key character varying(128),
    ebpp_value character varying(128)
);


ALTER TABLE public.ezc_buss_partner_params OWNER TO postgres;

--
-- TOC entry 219 (class 1259 OID 17878)
-- Name: ezc_buss_partner_xml_params; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_buss_partner_xml_params (
    ebpx_client numeric(3,0),
    ebpx_buss_partner character varying(10),
    ebpx_xml_transaction_id character varying(18),
    ebpx_xml_standard_in character varying(6),
    ebpx_xml_standard_out character varying(6)
);


ALTER TABLE public.ezc_buss_partner_xml_params OWNER TO postgres;

--
-- TOC entry 220 (class 1259 OID 17881)
-- Name: ezc_calendar; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_calendar (
    ecal_calendar_code character varying(6),
    ecal_effective_from timestamp without time zone NOT NULL,
    ecal_non_working_days character varying(2048),
    ecal_work_week character varying(2048),
    ecal_number_hours_day numeric(5,2),
    ecal_start_time character varying(10),
    ecal_break_time character varying(10),
    ecal_break_end character varying(10),
    ecal_active_yes_or_no character varying(1),
    ecal_effective_to timestamp without time zone
);


ALTER TABLE public.ezc_calendar OWNER TO postgres;

--
-- TOC entry 221 (class 1259 OID 17886)
-- Name: ezc_card_transactions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_card_transactions (
    ect_user_id character varying(20),
    ect_po_no character varying(35),
    ect_amount character varying(25),
    ect_tr_date timestamp without time zone,
    ect_sold_to character varying(10),
    ect_ship_to character varying(10),
    ect_status_code character varying(10),
    ect_tr_id character varying(30),
    ect_tr_message character varying(100),
    ect_tr_ext1 character varying(20),
    ect_tr_ext2 character varying(20),
    ect_tr_ext3 character varying(20)
);


ALTER TABLE public.ezc_card_transactions OWNER TO postgres;

--
-- TOC entry 222 (class 1259 OID 17889)
-- Name: ezc_cat_area_defaults; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_cat_area_defaults (
    ecad_client character varying(5),
    ecad_sys_key character varying(18),
    ecad_key character varying(18),
    ecad_value character varying(128)
);


ALTER TABLE public.ezc_cat_area_defaults OWNER TO postgres;

--
-- TOC entry 223 (class 1259 OID 17892)
-- Name: ezc_cat_attr_set; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_cat_attr_set (
    ecas_category_code character varying(20),
    ecas_attr_set character varying(20)
);


ALTER TABLE public.ezc_cat_attr_set OWNER TO postgres;

--
-- TOC entry 224 (class 1259 OID 17895)
-- Name: ezc_catalog_categories; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_catalog_categories (
    ecc_catalog_id character varying(20),
    ecc_category_id character varying(20)
);


ALTER TABLE public.ezc_catalog_categories OWNER TO postgres;

--
-- TOC entry 225 (class 1259 OID 17898)
-- Name: ezc_catalog_group; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_catalog_group (
    ecg_catalog_no numeric(8,0),
    ecg_sys_key character varying(18),
    ecg_product_group character varying(18),
    ecg_index_indicator character varying(1)
);


ALTER TABLE public.ezc_catalog_group OWNER TO postgres;

--
-- TOC entry 226 (class 1259 OID 17901)
-- Name: ezc_categories; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_categories (
    ec_code character varying(20),
    ec_status character varying(1),
    ec_parent character varying(20),
    ec_visible character varying(1),
    ec_image character varying(256),
    ec_thumb character varying(256),
    ec_sort numeric(5,2),
    ec_catalog_id character varying(5),
    ec_mat_count character varying(20)
);


ALTER TABLE public.ezc_categories OWNER TO postgres;

--
-- TOC entry 227 (class 1259 OID 17906)
-- Name: ezc_category_assets; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_category_assets (
    eca_category_code character varying(20),
    eca_asset_id character varying(50),
    eca_image_type character varying(4),
    eca_screen_name character varying(200),
    eca_catalog_id character varying(5)
);


ALTER TABLE public.ezc_category_assets OWNER TO postgres;

--
-- TOC entry 228 (class 1259 OID 17909)
-- Name: ezc_category_description; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_category_description (
    ecd_code character varying(20),
    ecd_lang character varying(3),
    ecd_desc character varying(80),
    ecd_text text,
    ecd_catalog_id character varying(5),
    ecd_profit_center character varying(20)
);


ALTER TABLE public.ezc_category_description OWNER TO postgres;

--
-- TOC entry 229 (class 1259 OID 17914)
-- Name: ezc_category_products; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_category_products (
    ecp_category_code character varying(20),
    ecp_product_code character varying(20),
    ecp_sort numeric(4,0),
    ecp_catalog_id character varying(5)
);


ALTER TABLE public.ezc_category_products OWNER TO postgres;

--
-- TOC entry 230 (class 1259 OID 17917)
-- Name: ezc_certificate_of_analysis; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_certificate_of_analysis (
    ezca_document_no character varying(10),
    ezca_item_no character varying(5),
    ezca_line_no character varying(3),
    ezca_arno character varying(16),
    ezca_doa timestamp without time zone,
    ezca_domfg timestamp without time zone,
    ezca_boxes character varying(16),
    ezca_spec_no character varying(16),
    ezca_ext1 character varying(20),
    ezca_ext2 character varying(20)
);


ALTER TABLE public.ezc_certificate_of_analysis OWNER TO postgres;

--
-- TOC entry 231 (class 1259 OID 17920)
-- Name: ezc_change_order_header; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_change_order_header (
    ecoh_doc_id integer NOT NULL,
    ecoh_po_ref character varying(50),
    ecoh_order_ref character varying(50),
    ecoh_status character varying(15),
    ecoh_modified_by character varying(50),
    ecoh_modified_on timestamp without time zone,
    ecoh_csr_name character varying(50),
    ecoh_csr_email character varying(50),
    ecoh_ext1 character varying(10),
    ecoh_ext2 character varying(10),
    ecoh_ext3 character varying(10)
);


ALTER TABLE public.ezc_change_order_header OWNER TO postgres;

--
-- TOC entry 232 (class 1259 OID 17923)
-- Name: ezc_change_order_items; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_change_order_items (
    ecoi_doc_id integer,
    ecoi_order_ref character varying(10),
    ecoi_line_item character varying(10),
    ecoi_change_type character varying(5),
    ecoi_sku character varying(20),
    ecoi_quantity character varying(10),
    ecoi_uom character varying(5),
    ecoi_changed_sku character varying(20),
    ecoi_changed_customer_sku character varying(20),
    ecoi_changed_quantity character varying(10),
    ecoi_changed_uom character varying(5),
    ecoi_cancel_reason character varying(20),
    ecoi_ext1 character varying(200),
    ecoi_ext2 character varying(10),
    ecoi_ext3 character varying(10)
);


ALTER TABLE public.ezc_change_order_items OWNER TO postgres;

--
-- TOC entry 233 (class 1259 OID 17926)
-- Name: ezc_claim_policies; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_claim_policies (
    ecp_title character varying(255),
    ecp_category character varying(255),
    ecp_created_at timestamp without time zone NOT NULL,
    ecp_updated_at timestamp without time zone NOT NULL,
    ecp_brand_id integer,
    ecp_id integer,
    ecp_pdf_ind bit(1) NOT NULL,
    ecp_xls_ind bit(1) NOT NULL
);


ALTER TABLE public.ezc_claim_policies OWNER TO postgres;

--
-- TOC entry 234 (class 1259 OID 17931)
-- Name: ezc_classification_assets; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_classification_assets (
    eca_classification_code character varying(20),
    eca_asset_id character varying(20),
    eca_image_type character varying(4),
    eca_screen_name character varying(200)
);


ALTER TABLE public.ezc_classification_assets OWNER TO postgres;

--
-- TOC entry 235 (class 1259 OID 17934)
-- Name: ezc_classification_description; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_classification_description (
    ecld_code character varying(20),
    ecld_lang character varying(3),
    ecld_desc character varying(50),
    ecld_text text
);


ALTER TABLE public.ezc_classification_description OWNER TO postgres;

--
-- TOC entry 236 (class 1259 OID 17939)
-- Name: ezc_classification_products; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_classification_products (
    ecp_classification_code character varying(20),
    ecp_product_code character varying(20)
);


ALTER TABLE public.ezc_classification_products OWNER TO postgres;

--
-- TOC entry 237 (class 1259 OID 17942)
-- Name: ezc_cnet_price; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_cnet_price (
    ecp_material character varying(40),
    ecp_mfr_name character varying(255),
    ecp_mfr_no character varying(40),
    ecp_price numeric(15,3),
    ecp_future_price numeric(15,3),
    ecp_effective_date timestamp without time zone,
    ecp_date timestamp without time zone
);


ALTER TABLE public.ezc_cnet_price OWNER TO postgres;

--
-- TOC entry 238 (class 1259 OID 17945)
-- Name: ezc_comm_archives; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_comm_archives (
    eca_title character varying(255),
    eca_category character varying(255),
    eca_created_at timestamp without time zone NOT NULL,
    eca_updated_at timestamp without time zone NOT NULL,
    eca_brand_id integer NOT NULL,
    eca_id integer NOT NULL,
    eca_pdf_ind bit(1) NOT NULL
);


ALTER TABLE public.ezc_comm_archives OWNER TO postgres;

--
-- TOC entry 239 (class 1259 OID 18007)
-- Name: ezc_dataconv_sapmatdataholder; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_dataconv_sapmatdataholder (
    logged_on timestamp without time zone NOT NULL,
    material character varying(20),
    matdesc character varying(42),
    "HIERARCHY" character varying(18),
    uofmeasure character varying(3),
    stdprice numeric(8,3),
    currency character varying(5),
    delflag character varying(1),
    quantity numeric(10,3),
    specs character varying(40),
    upcno character varying(18),
    item_cat_group character varying(4),
    material_group1 character varying(3),
    material_group2 character varying(3),
    material_group3 character varying(3),
    material_group4 character varying(3),
    material_group5 character varying(3),
    sap_comm_group character varying(2),
    soip_category character varying(3),
    xdch_status character varying(2),
    mat_pric_group character varying(2),
    division character varying(2),
    volum numeric(10,3),
    def_del_plant character varying(4),
    sales_org character varying(4),
    dist_channel character varying(4),
    reserved_1 character varying(40),
    reserved_2 character varying(40),
    reserved_3 character varying(40),
    reserved_4 character varying(40),
    reserved_5 character varying(40),
    reserved_6 character varying(200),
    reserved_7 character varying(200),
    reserved_8 character varying(200),
    reserved_9 text,
    reserved_10 text,
    reserved_11 text
);


ALTER TABLE public.ezc_dataconv_sapmatdataholder OWNER TO postgres;

--
-- TOC entry 240 (class 1259 OID 18417)
-- Name: ezc_product_classification; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_product_classification (
    epcl_code character varying(20),
    epcl_status character varying(1),
    epcl_type character varying(1),
    epcl_visible character varying(1),
    epcl_image bytea,
    epcl_thumb bytea
);


ALTER TABLE public.ezc_product_classification OWNER TO postgres;

--
-- TOC entry 241 (class 1259 OID 18422)
-- Name: ezc_product_descriptions; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_product_descriptions (
    epd_product_code character varying(20),
    epd_lang_code character varying(6),
    epd_product_desc character varying(200),
    epd_product_details text,
    epd_product_prop1 character varying(50),
    epd_product_prop2 character varying(50),
    epd_product_prop3 character varying(50),
    epd_product_prop4 character varying(50),
    epd_product_prop5 character varying(50),
    epd_product_prop6 character varying(50),
    epd_catalog_id character varying(5)
);


ALTER TABLE public.ezc_product_descriptions OWNER TO postgres;

--
-- TOC entry 242 (class 1259 OID 18427)
-- Name: ezc_product_descriptions_0325222; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_product_descriptions_0325222 (
    epd_product_code character varying(20),
    epd_lang_code character varying(6),
    epd_product_desc character varying(200),
    epd_product_details text,
    epd_product_prop1 character varying(50),
    epd_product_prop2 character varying(50),
    epd_product_prop3 character varying(50),
    epd_product_prop4 character varying(50),
    epd_product_prop5 character varying(50),
    epd_product_prop6 character varying(50),
    epd_catalog_id character varying(5)
);


ALTER TABLE public.ezc_product_descriptions_0325222 OWNER TO postgres;

--
-- TOC entry 243 (class 1259 OID 18432)
-- Name: ezc_product_group; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_product_group (
    epg_no character varying(18),
    epg_sys_no numeric(3,0),
    epg_group_level numeric(3,0),
    epg_deletion_flag character varying(1),
    epg_global_view_flag character varying(1),
    epg_gif_flag character varying(1),
    epg_no_of_items numeric(3,0),
    epg_terminal_flag character varying(1)
);


ALTER TABLE public.ezc_product_group OWNER TO postgres;

--
-- TOC entry 244 (class 1259 OID 18435)
-- Name: ezc_product_group_desc; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_product_group_desc (
    epg_no character varying(18),
    epgd_lang character varying(2),
    epgd_desc character varying(120),
    epgd_web_desc character varying(120)
);


ALTER TABLE public.ezc_product_group_desc OWNER TO postgres;

--
-- TOC entry 245 (class 1259 OID 18438)
-- Name: ezc_product_relations; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_product_relations (
    epr_product_code1 character varying(20),
    epr_product_code2 character varying(20),
    epr_relation_type character varying(4),
    epr_rel_qty numeric(13,3)
);


ALTER TABLE public.ezc_product_relations OWNER TO postgres;

--
-- TOC entry 246 (class 1259 OID 18441)
-- Name: ezc_products; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_products (
    ezp_product_code character varying(20),
    ezp_type character varying(5),
    ezp_sub_type character varying(20),
    ezp_status character varying(4),
    ezp_web_sku character varying(18),
    ezp_web_prod_id character varying(40),
    ezp_upc_code character varying(20),
    ezp_erp_code character varying(40),
    ezp_brand character varying(20),
    ezp_family character varying(20),
    ezp_model character varying(20),
    ezp_color character varying(20),
    ezp_finish character varying(20),
    ezp_size character varying(40),
    ezp_length character varying(20),
    ezp_width character varying(20),
    ezp_length_uom character varying(10),
    ezp_weight character varying(10),
    ezp_weight_uom character varying(10),
    ezp_volume character varying(40),
    ezp_volume_uom character varying(10),
    ezp_style character varying(40),
    ezp_new_from timestamp without time zone,
    ezp_new_to timestamp without time zone,
    ezp_curr_price numeric(15,3),
    ezp_curr_eff_date timestamp without time zone,
    ezp_future_price numeric(15,3),
    ezp_future_eff_date timestamp without time zone,
    ezp_featured character varying(10),
    ezp_discontinued character varying(1),
    ezp_discontinue_date timestamp without time zone,
    ezp_replaces_item character varying(20),
    ezp_alternate1 character varying(20),
    ezp_alternate2 character varying(20),
    ezp_alternate3 character varying(20),
    ezp_attr1 character varying(20),
    ezp_attr2 character varying(200),
    ezp_attr3 character varying(200),
    ezp_attr4 character varying(20),
    ezp_attr5 character varying(20),
    ezp_luxury character varying(50),
    ezp_catalog_id character varying(5),
    ezp_mat_pric_group character varying(2),
    ezp_profit_center character varying(10),
    ezp_serial_profile character varying(4),
    ezp_batch_managed character varying(1),
    ezp_gross_weight character varying(10),
    ezp_item_cat character varying(10),
    ezp_sort integer,
    ezp_category character varying(100),
    ezp_sub_category character varying(100)
);


ALTER TABLE public.ezc_products OWNER TO postgres;

--
-- TOC entry 247 (class 1259 OID 18446)
-- Name: ezc_profit_center; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_profit_center (
    epc_category_code character varying(20),
    epc_profit_center character varying(20),
    epc_catalog_id character varying(5)
);


ALTER TABLE public.ezc_profit_center OWNER TO postgres;

--
-- TOC entry 248 (class 1259 OID 18449)
-- Name: ezc_projection_header; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_projection_header (
    ezpr_projectionid character varying(10),
    ezpr_userid character varying(10),
    ezpr_usertype character varying(10),
    ezpr_doe timestamp without time zone,
    ezpr_period_start_date timestamp without time zone,
    ezpr_period_end_date timestamp without time zone,
    ezpr_periods numeric(2,0),
    ezpr_syskey character varying(18),
    ezpr_soldto character varying(10),
    ezpr_additional1 character varying(30),
    ezpr_additional2 character varying(30),
    ezpr_additional3 character varying(30)
);


ALTER TABLE public.ezc_projection_header OWNER TO postgres;

--
-- TOC entry 249 (class 1259 OID 18452)
-- Name: ezc_projection_lines; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_projection_lines (
    ezpr_projectionid character varying(10),
    ezpr_status character varying(1),
    ezpr_product character varying(18),
    ezpr_pack character varying(5),
    ezpr_values character varying(256),
    ezpr_prices character varying(256)
);


ALTER TABLE public.ezc_projection_lines OWNER TO postgres;

--
-- TOC entry 250 (class 1259 OID 18457)
-- Name: ezc_promotional_codes; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_promotional_codes (
    epc_code character varying(20),
    epc_number character varying(15),
    epc_promo_type character varying(10),
    epc_mfr_id character varying(30),
    epc_prod_cat character varying(40),
    epc_discount numeric(15,2),
    epc_created_by character varying(20),
    epc_created_on timestamp without time zone,
    epc_modified_by character varying(20),
    epc_modified_on timestamp without time zone,
    epc_valid_from timestamp without time zone,
    epc_valid_to timestamp without time zone,
    epc_status character varying(10),
    epc_ext1 character varying(100),
    epc_ext2 character varying(250),
    epc_ext3 text,
    epc_syskey character varying(6)
);


ALTER TABLE public.ezc_promotional_codes OWNER TO postgres;

--
-- TOC entry 251 (class 1259 OID 18462)
-- Name: ezc_qcf_comments; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_qcf_comments (
    eqc_code character varying(10),
    eqc_comment_no character varying(10),
    eqc_user character varying(50),
    eqc_date timestamp without time zone,
    eqc_comments character varying(2000),
    eqc_ext1 character varying(100),
    eqc_ext2 character varying(250),
    eqc_ext3 text,
    eqc_dest_user character varying(50),
    eqc_type character varying(10),
    eqc_query_map character varying(10)
);


ALTER TABLE public.ezc_qcf_comments OWNER TO postgres;

--
-- TOC entry 252 (class 1259 OID 18467)
-- Name: ezc_registration_form_dtls; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_registration_form_dtls (
    erfd_reg_id integer NOT NULL,
    erfd_first_name character varying(30),
    erfd_last_name character varying(30),
    erfd_email character varying(60),
    erfd_phone_no character varying(15),
    erfd_company character varying(60),
    erfd_street character varying(60),
    erfd_city character varying(60),
    erfd_state character varying(10),
    erfd_zip character varying(10),
    erfd_sales_manager character varying(60),
    erfd_gatekeeper character varying(1),
    erfd_gatekeeper_man character varying(60),
    erfd_gatekeeper_man_phone character varying(15),
    erfd_gatekeeper_man_title character varying(60),
    erfd_gatekeeper_man_email character varying(60),
    erfd_requested_on timestamp without time zone,
    erfd_ext1 character varying(100),
    erfd_ext2 character varying(100),
    erfd_ext3 character varying(100),
    erfd_country character varying(4),
    erfd_customer_number character varying(10),
    erfd_user_id character varying(20),
    erfd_partner_id character varying(20),
    erfd_comments character varying(400),
    erfd_sales_man_id character varying(15),
    erfd_email_text character varying(100)
);


ALTER TABLE public.ezc_registration_form_dtls OWNER TO postgres;

--
-- TOC entry 253 (class 1259 OID 18472)
-- Name: ezc_rep_customer_sync; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_rep_customer_sync (
    ercs_sync_date timestamp without time zone,
    ercs_sales_group character varying(5),
    ercs_sold_to character varying(10),
    ercs_syskey character varying(6)
);


ALTER TABLE public.ezc_rep_customer_sync OWNER TO postgres;

--
-- TOC entry 254 (class 1259 OID 18475)
-- Name: ezc_report_exec_store; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_report_exec_store (
    ers_counter numeric(5,0),
    ers_report_no numeric(10,0),
    ers_spool_no character varying(20),
    ers_system_no numeric(3,0),
    ers_user_id character varying(10),
    ers_creation_date character varying(10),
    ers_creation_time character varying(10),
    ers_report_path character varying(128),
    ers_view_flag character varying(1),
    ers_email character varying(128),
    ers_report_format character varying(1),
    ers_ext1 character varying(20),
    ers_ext2 character varying(20)
);


ALTER TABLE public.ezc_report_exec_store OWNER TO postgres;

--
-- TOC entry 255 (class 1259 OID 18478)
-- Name: ezc_report_info; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_report_info (
    eri_report_no numeric(10,0),
    eri_system_no numeric(3,0),
    eri_report_name character varying(64),
    eri_report_desc character varying(128),
    eri_lang character varying(2),
    eri_report_type numeric(3,0),
    eri_exec_type character varying(1),
    eri_visible_level character varying(1),
    eri_business_domain character varying(1),
    eri_report_status character varying(1),
    eri_ext1 character varying(20),
    eri_ext2 character varying(20)
);


ALTER TABLE public.ezc_report_info OWNER TO postgres;

--
-- TOC entry 256 (class 1259 OID 18481)
-- Name: ezc_report_params; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_report_params (
    erp_report_no numeric(10,0),
    erp_param_no numeric(3,0),
    erp_param_name character varying(18),
    erp_param_desc character varying(128),
    erp_param_type character varying(1),
    erp_data_type character varying(128),
    erp_length numeric(5,0),
    erp_is_mandatory character varying(1),
    erp_is_customer character varying(1),
    erp_is_hidden character varying(1),
    erp_chk_defaults character varying(1),
    erp_method_name character varying(40),
    erp_ext1 character varying(20),
    erp_ext2 character varying(20)
);


ALTER TABLE public.ezc_report_params OWNER TO postgres;

--
-- TOC entry 257 (class 1259 OID 18484)
-- Name: ezc_report_values; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_report_values (
    erv_report_no numeric(10,0),
    erv_param_no numeric(3,0),
    erv_param_value character varying(64),
    erv_param_value_high character varying(64),
    erv_retrieval_mode character varying(1),
    erv_operator character varying(2),
    erv_ext1 character varying(20),
    erv_ext2 character varying(20)
);


ALTER TABLE public.ezc_report_values OWNER TO postgres;

--
-- TOC entry 258 (class 1259 OID 18487)
-- Name: ezc_role_auth; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_role_auth (
    era_role_nr character varying(16),
    era_sys_no numeric(3,0),
    era_auth_key character varying(16),
    era_auth_value character varying(128),
    era_actions_or_statuses character varying(1024)
);


ALTER TABLE public.ezc_role_auth OWNER TO postgres;

--
-- TOC entry 259 (class 1259 OID 18492)
-- Name: ezc_roles_by_user; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_roles_by_user (
    erbu_user_id character varying(20),
    erbu_system_key numeric(3,0),
    erbu_role numeric(3,0)
);


ALTER TABLE public.ezc_roles_by_user OWNER TO postgres;

--
-- TOC entry 260 (class 1259 OID 18495)
-- Name: ezc_saledoc_mails; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_saledoc_mails (
    esm_product_code character varying(18),
    esm_to character varying(1024),
    esm_cc character varying(1024),
    esm_edd character varying(1024),
    esm_plant character varying(10)
);


ALTER TABLE public.ezc_saledoc_mails OWNER TO postgres;

--
-- TOC entry 261 (class 1259 OID 18500)
-- Name: ezc_sales_discounts; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_sales_discounts (
    esd_disc_no character varying(10),
    esd_syskey character varying(6),
    esd_disc_type character varying(10),
    esd_mfr_id character varying(30),
    esd_prod_cat character varying(40),
    esd_customer character varying(10),
    esd_discount numeric(5,2),
    esd_created_by character varying(20),
    esd_created_on timestamp without time zone,
    esd_modified_by character varying(20),
    esd_modified_on timestamp without time zone,
    esd_valid_from timestamp without time zone,
    esd_valid_to timestamp without time zone,
    esd_status character varying(10),
    esd_ext1 character varying(100),
    esd_ext2 character varying(250),
    esd_ext3 text
);


ALTER TABLE public.ezc_sales_discounts OWNER TO postgres;

--
-- TOC entry 262 (class 1259 OID 18505)
-- Name: ezc_sales_doc_header; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_sales_doc_header (
    esdh_doc_number character varying(10),
    esdh_sys_key character varying(18),
    esdh_doc_type character varying(4),
    esdh_create_on timestamp without time zone NOT NULL,
    esdh_created_by character varying(20),
    esdh_modified_by character varying(20),
    esdh_modified_on timestamp without time zone,
    esdh_back_end_order character varying(18),
    esdh_order_date timestamp without time zone,
    esdh_transfer_date timestamp without time zone,
    esdh_status_date timestamp without time zone,
    esdh_res1 character varying(12),
    esdh_res2 character varying(3072),
    esdh_del_flag character varying(1),
    esdh_collect_no character varying(10),
    esdh_sales_org character varying(4),
    esdh_distr_chan character varying(2),
    esdh_division character varying(2),
    esdh_sales_grp character varying(3),
    esdh_sales_off character varying(4),
    esdh_req_date_h timestamp without time zone,
    esdh_date_type character varying(1),
    esdh_doc_currency character varying(6),
    esdh_net_value numeric(16,6),
    esdh_po_no character varying(35),
    esdh_ref_doc_no character varying(10),
    esdh_purch_no character varying(20),
    esdh_purch_date timestamp without time zone,
    esdh_po_method character varying(100),
    esdh_po_supplem character varying(4),
    esdh_ref_1 character varying(20),
    esdh_name character varying(30),
    esdh_telephone character varying(16),
    esdh_price_grp character varying(2),
    esdh_cust_group character varying(2),
    esdh_sales_dist character varying(6),
    esdh_price_list character varying(2),
    esdh_incoterms1 character varying(3),
    esdh_incoterms2 character varying(28),
    esdh_pmnttrms character varying(4),
    esdh_dlv_block character varying(2),
    esdh_bill_block character varying(2),
    esdh_ord_reason character varying(3),
    esdh_compl_dlv character varying(1),
    esdh_price_date timestamp without time zone,
    esdh_qt_valid_f timestamp without time zone,
    esdh_qt_valid_t timestamp without time zone,
    esdh_ct_valid_f timestamp without time zone,
    esdh_ct_valid_t timestamp without time zone,
    esdh_cust_grp1 character varying(3),
    esdh_cust_grp2 character varying(3),
    esdh_cust_grp3 character varying(3),
    esdh_cust_grp4 character varying(3),
    esdh_cust_grp5 character varying(10),
    esdh_status character varying(15),
    esdh_sold_to character varying(10),
    esdh_soldto_addr_1 character varying(64),
    esdh_soldto_addr_2 character varying(64),
    esdh_soldto_addr_3 character varying(64),
    esdh_soldto_country character varying(3),
    esdh_soldto_state character varying(64),
    esdh_soldto_pin character varying(64),
    esdh_ship_to character varying(10),
    esdh_shipto_addr_1 character varying(100),
    esdh_shipto_addr_2 character varying(200),
    esdh_shipto_addr_3 character varying(64),
    esdh_shipto_country character varying(3),
    esdh_shipto_state character varying(64),
    esdh_shipto_pin character varying(64),
    esdh_text1 text,
    esdh_text2 character varying(1024),
    esdh_ref_doc character varying(40),
    esdh_agent_code character varying(10),
    esdh_type character varying(1),
    esdh_discount_cash numeric(10,2),
    esdh_discount_percentage numeric(2,0),
    edsh_freight character varying(10),
    edsh_text3 character varying(1024),
    esdh_class2 character varying(1),
    esdh_text4 character varying(1024),
    esdh_sap_so character varying(10),
    esdh_promo_code character varying(20),
    esdh_freight_weight numeric(15,3),
    esdh_freight_price character varying(15),
    esdh_freight_ins character varying(15),
    esdh_ship_method character varying(10),
    esdh_carrier_acc character varying(10),
    esdh_billto_name character varying(20),
    esdh_billto_addr1 character varying(60),
    esdh_billto_addr2 character varying(200),
    esdh_billto_street character varying(20),
    esdh_billto_city character varying(40),
    esdh_billto_state character varying(5),
    esdh_billto_pin character varying(10),
    esdh_billto_phone character varying(16),
    esdh_template_name character varying(20),
    esdh_save_flag character varying(10),
    esdh_toact character varying(40),
    esdh_actby character varying(20),
    esdh_defcat_l1 character varying(40),
    esdh_defcat_l2 character varying(40),
    esdh_defcat_l3 character varying(80)
);


ALTER TABLE public.ezc_sales_doc_header OWNER TO postgres;

--
-- TOC entry 263 (class 1259 OID 18510)
-- Name: ezc_sales_doc_items; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_sales_doc_items (
    esdi_sales_doc character varying(10),
    esdi_sys_key character varying(18),
    esdi_sales_doc_item numeric(6,0),
    esdi_material character varying(18),
    esdi_pricing_ref_mat character varying(18),
    esdi_batch_no character varying(10),
    esdi_material_group character varying(50),
    esdi_short_text character varying(100),
    esdi_req_date timestamp without time zone,
    esdi_promise_date timestamp without time zone,
    esdi_desired_price numeric(20,10),
    esdi_commited_price numeric(20,10),
    esdi_del_flag character varying(1),
    esdi_item_category character varying(4),
    esdi_item_type character varying(10),
    esdi_relevent_for_delivry character varying(1),
    esdi_relevant_for_billing character varying(1),
    esdi_reason_for_rejection character varying(2),
    esdi_product_group character varying(18),
    esdi_outline_agr_val numeric(13,2),
    esdi_qty_in_sales_unit numeric(13,0),
    esdi_sales_unit character varying(3),
    esdi_conv_factor_su_to_bu numeric(5,0),
    esdi_base_unit_of_measure character varying(3),
    esdi_qty_in_base_unit numeric(13,0),
    esdi_item_no_of_customer numeric(6,0),
    esdi_customer_mat character varying(18),
    esdi_division character varying(2),
    esdi_buss_area character varying(4),
    esdi_net_val_of_order numeric(15,2),
    esdi_doc_currency character varying(5),
    esdi_batch_split_allowed character varying(1),
    esdi_req_qty numeric(15,3),
    esdi_confirmed_qty numeric(15,3),
    esdi_delivery_priority_plant character varying(4),
    esdi_plant character varying(4),
    esdi_storage_loc character varying(4),
    esdi_shipping_point character varying(4),
    esdi_date_record_created character varying(10),
    esdi_created_by character varying(20),
    esdi_net_price numeric(15,3),
    esdi_cash_discount_indicator character varying(1),
    esdi_avail_check_group character varying(2),
    esdi_pricing_group character varying(2),
    esdi_account_assign_group character varying(2),
    esdi_reason_mat_substitution character varying(40),
    esdi_statistical_values character varying(1),
    esdi_statistics_dated character varying(10),
    esdi_buss_transaction_type character varying(4),
    esdi_preference_ind_exp_imp character varying(1),
    esdi_mat_freight_group character varying(8),
    esdi_foc numeric(2,0),
    esdi_ref_doc_item numeric(6,0),
    esdi_notes text,
    esdi_back_end_order character varying(18),
    esdi_sales_org character varying(4),
    esdi_distr_chan character varying(2),
    esdi_back_end_item character varying(6),
    esdi_ship_to character varying(10),
    esdi_dlv_block character varying(1),
    esdi_incoterms1 character varying(3),
    esdi_incoterms2 character varying(28),
    esdi_remarks character varying(1024),
    esdi_list_price numeric(20,10),
    esdi_sap_price numeric(20,10),
    esdi_disc_code character varying(10),
    esdi_promo_code character varying(20),
    esdi_freight_weight numeric(15,3),
    esdi_freight_ins character varying(15),
    esdi_vip_flag character varying(1),
    esdi_display_flag character varying(1),
    esdi_quickship_flag character varying(1),
    esdi_quote_ref_no character varying(10),
    esdi_quote_line_no character varying(6),
    esdi_points character varying(10),
    esdi_points_group character varying(40),
    esdi_cust_sku character varying(20),
    esdi_cust_po_lineno character varying(20),
    esdi_ref_question_no character varying(5),
    esdi_order_type character varying(10),
    esdi_item_multiplier character varying(20),
    esdi_item_upc character varying(20),
    esdi_listprice character varying(10)
);


ALTER TABLE public.ezc_sales_doc_items OWNER TO postgres;

--
-- TOC entry 264 (class 1259 OID 18515)
-- Name: ezc_sales_doc_partners; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_sales_doc_partners (
    esdp_sales_doc character varying(10),
    esdp_sales_area character varying(18),
    esdp_partner_function character varying(4),
    esdp_partner_number character varying(10),
    esdp_ezc_customer_number character varying(10),
    esdp_erp_customer_number character varying(10),
    esdp_itm_number numeric(6,0),
    esdp_title character varying(15),
    esdp_name character varying(35),
    esdp_name_2 character varying(35),
    esdp_name_3 character varying(35),
    esdp_name_4 character varying(35),
    esdp_street character varying(35),
    esdp_country character varying(3),
    esdp_postl_code character varying(10),
    esdp_city character varying(35),
    esdp_district character varying(35),
    esdp_region character varying(3),
    esdp_po_box character varying(10),
    esdp_telephone character varying(16),
    esdp_telephone2 character varying(16),
    esdp_fax_number character varying(31),
    esdp_teletex_no character varying(30),
    esdp_telex_no character varying(30),
    esdp_unload_pt character varying(25),
    esdp_transpzone character varying(10),
    esdp_taxjurcode character varying(15),
    esdp_back_end_number character varying(10),
    esdp_back_itm_number character varying(6)
);


ALTER TABLE public.ezc_sales_doc_partners OWNER TO postgres;

--
-- TOC entry 265 (class 1259 OID 18520)
-- Name: ezc_shipping_claims_comments; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_shipping_claims_comments (
    escc_doc_id character varying(20),
    escc_comments character varying(3000),
    escc_user_id character varying(100),
    escc_date timestamp without time zone NOT NULL,
    escc_ext1 character varying(500),
    escc_ext2 character varying(50),
    escc_visibility character varying(50)
);


ALTER TABLE public.ezc_shipping_claims_comments OWNER TO postgres;

--
-- TOC entry 266 (class 1259 OID 18525)
-- Name: ezc_shipping_claims_header; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_shipping_claims_header (
    esch_doc_id character varying(200),
    esch_type character varying(200),
    esch_doc_number character varying(200),
    esch_component character varying(200),
    esch_date character varying(200),
    esch_claim_issue character varying(200),
    esch_original_product character varying(200),
    esch_original_quantity character varying(200),
    esch_resolution character varying(200),
    esch_serial_number character varying(200),
    esch_contact_no character varying(80),
    esch_add_comments character varying(800),
    esch_admin_comments character varying(800),
    esch_shipto_name character varying(100),
    esch_shipto_addr character varying(200),
    esch_shipto_city character varying(400),
    esch_shipto_zip character varying(400),
    esch_shipto_state character varying(400),
    esch_shipto_country character varying(400),
    esch_soldto_no character varying(100),
    esch_soldto_name character varying(100),
    esch_status character varying(100),
    esch_other_issue character varying(200),
    esch_po_number character varying(200),
    esch_created_by character varying(100),
    esch_created_on timestamp without time zone,
    esch_modified_by character varying(100),
    esch_modified_on timestamp without time zone,
    esch_ext1 character varying(50),
    esch_ext2 character varying(50),
    esch_ext3 character varying(50),
    esch_ext4 character varying(50),
    esch_ext5 character varying(50),
    esch_ext6 character varying(50)
);


ALTER TABLE public.ezc_shipping_claims_header OWNER TO postgres;

--
-- TOC entry 267 (class 1259 OID 18530)
-- Name: ezc_shipping_claims_products; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_shipping_claims_products (
    escp_doc_id character varying(200),
    escp_product character varying(200),
    escp_product_desc character varying(200),
    escp_quantity character varying(200)
);


ALTER TABLE public.ezc_shipping_claims_products OWNER TO postgres;

--
-- TOC entry 268 (class 1259 OID 18535)
-- Name: ezc_shopping_cart; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_shopping_cart (
    esc_user_id character varying(20),
    esc_sys_key character varying(6),
    esc_mat_no character varying(20),
    esc_mat_desc character varying(200),
    esc_line_no character varying(6),
    esc_req_qty numeric(15,3),
    esc_req_date character varying(20),
    esc_ven_cat character varying(40),
    esc_uom character varying(3),
    esc_unit_price numeric(15,3),
    esc_brand character varying(20),
    esc_mat_status character varying(4),
    esc_quote_cust character varying(10),
    esc_job_quote character varying(10),
    esc_job_item character varying(6),
    esc_my_sku character varying(20),
    esc_my_po_line character varying(20),
    esc_img_url character varying(200),
    esc_ean_upc character varying(20),
    esc_prog_type character varying(10),
    esc_ord_type character varying(10),
    esc_comm_grp character varying(10),
    esc_sales_org character varying(10),
    esc_dist_chnl character varying(10),
    esc_division character varying(10),
    esc_volume character varying(10),
    esc_points character varying(10),
    esc_kit_comp character varying(10),
    esc_weight character varying(18),
    esc_weight_uom character varying(10),
    esc_ext1 character varying(10),
    esc_ext2 character varying(400),
    esc_ext3 character varying(10),
    esc_promo_code character varying(10),
    esc_fd_category character varying(40),
    esc_fd_account character varying(10),
    esc_brand_site character varying(10),
    esc_sold_to_code character varying(10),
    esc_ship_to_code character varying(10),
    esc_ship_to_state character varying(10),
    esc_delv_group character varying(3)
);


ALTER TABLE public.ezc_shopping_cart OWNER TO postgres;

--
-- TOC entry 269 (class 1259 OID 18540)
-- Name: ezc_simple_triggers; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_simple_triggers (
    trigger_name character varying(80),
    trigger_group character varying(80),
    repeat_count bigint NOT NULL,
    repeat_interval bigint NOT NULL,
    times_triggered bigint NOT NULL
);


ALTER TABLE public.ezc_simple_triggers OWNER TO postgres;

--
-- TOC entry 270 (class 1259 OID 18543)
-- Name: ezc_site_globals; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_site_globals (
    esg_site_no numeric(3,0),
    esg_outside_catalog numeric(8,0),
    esg_audit_log_on_off_flag character varying(1),
    esg_sync_trace_flag character varying(1),
    esg_sch_time_mat_sync character varying(10),
    esg_sch_time_cust_sync character varying(10),
    esg_max_number numeric(6,0),
    esg_multiple_systems_allowed numeric(1,0),
    esg_multiple_sales_areas numeric(1,0),
    esg_max_buss_users numeric(6,0),
    esg_max_intranet_users numeric(6,0),
    esg_max_internet_users numeric(6,0),
    esg_unlimited_buss_users character varying(1),
    esg_unlimited_intranet_users character varying(1),
    esg_unlimited_internet_users character varying(1),
    esg_base_erp_sys_no numeric(3,0),
    ezg_cust_valid_flag character varying(1)
);


ALTER TABLE public.ezc_site_globals OWNER TO postgres;

--
-- TOC entry 271 (class 1259 OID 18546)
-- Name: ezc_snippets; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_snippets (
    ezs_id integer NOT NULL,
    ezs_snippet_type character varying(50),
    ezs_snippet_desc character varying(255),
    ezs_snippet_body character varying,
    ezs_created_at timestamp without time zone,
    ezs_updated_at timestamp without time zone
);


ALTER TABLE public.ezc_snippets OWNER TO postgres;

--
-- TOC entry 272 (class 1259 OID 18551)
-- Name: ezc_so_cancel_header; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_so_cancel_header (
    esch_id numeric(10,0),
    esch_po_num character varying(100),
    esch_so_num character varying(15),
    esch_created_by character varying(20),
    esch_created_on timestamp without time zone,
    esch_modified_by character varying(20),
    esch_modified_on timestamp without time zone,
    esch_status character varying(3),
    esch_type character varying(10),
    esch_ext1 character varying(10),
    esch_ext2 character varying(10),
    esch_ext3 character varying(10),
    esch_sold_to character varying(15),
    esch_syskey character varying(10),
    esch_reason character varying(50),
    esch_cust_text text,
    esch_contact_name character varying(50),
    esch_contact_email character varying(50),
    esch_contact_phone character varying(14),
    esch_sap_reason character varying(4),
    esch_inco_term1 character varying(4),
    esch_inco_term2 character varying(20),
    esch_shipping_partner character varying(10),
    esch_ship_to character varying(15),
    esch_ship_to_res character varying(3),
    esch_ship_to_name character varying(40),
    esch_ship_to_street1 character varying(40),
    esch_ship_to_street2 character varying(40),
    esch_ship_to_city character varying(40),
    esch_ship_to_state character varying(40),
    esch_ship_to_zip character varying(15),
    esch_ship_to_country character varying(3),
    esch_header_fees_type character varying(3),
    esch_header_fees_value character varying(13),
    esch_internal_text text,
    esch_approver_note text,
    esch_ship_to_phone character varying(20),
    esch_expire_on timestamp without time zone
);


ALTER TABLE public.ezc_so_cancel_header OWNER TO postgres;

--
-- TOC entry 273 (class 1259 OID 18556)
-- Name: ezc_so_cancel_items; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_so_cancel_items (
    esci_id numeric(10,0),
    esci_so_num character varying(15),
    esci_so_item character varying(10),
    esci_mat_code character varying(50),
    esci_mat_desc character varying(200),
    esci_quantity character varying(20),
    esci_rej_reason character varying(5),
    esci_comments character varying(500),
    esci_type character varying(10),
    esci_status character varying(3),
    esci_ext1 character varying(10),
    esci_ext2 character varying(10),
    esci_ext3 character varying(10),
    esci_ret_mat character varying(50),
    esci_ret_qty character varying(20),
    esci_retmat_np character varying(20),
    esci_so_sorg character varying(4),
    esci_so_dch character varying(2),
    esci_so_div character varying(2),
    esci_back_end_order character varying(10),
    esci_back_end_item character varying(6),
    esci_inv_num character varying(20),
    esci_inv_item character varying(6),
    esci_plant character varying(10),
    esci_req_qty character varying(20)
);


ALTER TABLE public.ezc_so_cancel_items OWNER TO postgres;

--
-- TOC entry 274 (class 1259 OID 18561)
-- Name: ezc_so_dyn_partner; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_so_dyn_partner (
    esdp_so_n0 character varying(10),
    esdp_ship_to_cust character varying(10)
);


ALTER TABLE public.ezc_so_dyn_partner OWNER TO postgres;

--
-- TOC entry 275 (class 1259 OID 18643)
-- Name: ezc_trigger_listeners; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_trigger_listeners (
    trigger_name character varying(200),
    trigger_group character varying(200),
    listener_name character varying(200)
);


ALTER TABLE public.ezc_trigger_listeners OWNER TO postgres;

--
-- TOC entry 276 (class 1259 OID 18648)
-- Name: ezc_triggers; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_triggers (
    trigger_name character varying(80),
    trigger_group character varying(80),
    job_name character varying(80),
    job_group character varying(80),
    is_volatile character varying(1),
    description character varying(120),
    next_fire_time bigint,
    prev_fire_time bigint,
    priority integer,
    trigger_state character varying(16),
    trigger_type character varying(8),
    start_time bigint NOT NULL,
    end_time bigint,
    calendar_name character varying(80),
    misfire_instr smallint,
    job_data bytea
);


ALTER TABLE public.ezc_triggers OWNER TO postgres;

--
-- TOC entry 277 (class 1259 OID 18653)
-- Name: ezc_upload_docs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_upload_docs (
    eud_upload_no numeric(6,0),
    eud_syskey character varying(6),
    eud_object_type character varying(10),
    eud_object_no character varying(20),
    eud_status character varying(16),
    eud_created_on timestamp without time zone,
    eud_created_by character varying(20)
);


ALTER TABLE public.ezc_upload_docs OWNER TO postgres;

--
-- TOC entry 278 (class 1259 OID 18656)
-- Name: ezc_uploaddoc_files; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_uploaddoc_files (
    euf_upload_no numeric(6,0),
    euf_type character varying(20),
    euf_client_file_name character varying(100),
    euf_server_file_name character varying(200)
);


ALTER TABLE public.ezc_uploaddoc_files OWNER TO postgres;

--
-- TOC entry 279 (class 1259 OID 18659)
-- Name: ezc_url_mapping; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_url_mapping (
    eum_short_url character varying(50),
    eum_actual_url character varying(300)
);


ALTER TABLE public.ezc_url_mapping OWNER TO postgres;

--
-- TOC entry 280 (class 1259 OID 18662)
-- Name: ezc_user_auth; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_user_auth (
    eua_user_id character varying(20),
    eua_sys_no numeric(3,0),
    eua_auth_key character varying(16),
    eua_auth_value character varying(128),
    eua_role_or_auth character varying(1)
);


ALTER TABLE public.ezc_user_auth OWNER TO postgres;

--
-- TOC entry 281 (class 1259 OID 18665)
-- Name: ezc_user_defaults; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_user_defaults (
    eud_user_id character varying(20),
    eud_sys_key character varying(18),
    eud_cust_no character varying(10),
    eud_key character varying(16),
    eud_value character varying(256),
    eud_default_flag character varying(1),
    eud_is_usera_key character varying(1)
);


ALTER TABLE public.ezc_user_defaults OWNER TO postgres;

--
-- TOC entry 282 (class 1259 OID 18668)
-- Name: ezc_user_groups; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_user_groups (
    eug_id numeric(3,0),
    eug_sys_no numeric(3,0),
    eug_name character varying(60),
    eug_validation_type numeric(1,0),
    eug_connect_type numeric(1,0),
    eug_history_read numeric(1,0),
    eug_material_access numeric(1,0),
    eug_cust_info_access numeric(1,0),
    eug_db_mem_cache numeric(1,0),
    eug_data_sync_type numeric(1,0),
    eug_history_write numeric(1,0),
    eug_r3_host character varying(32),
    eug_r3_sys_no numeric(3,0),
    eug_r3_gateway_host character varying(32),
    eug_r3_sys_name character varying(128),
    eug_r3_group_name character varying(64),
    eug_r3_msg_server character varying(64),
    eug_r3_load_balance character varying(1),
    eug_r3_check_auth character varying(1),
    eug_r3_code_page numeric(4,0),
    eug_r3_lang character varying(2),
    eug_r3_client character varying(3),
    eug_r3_user_id character varying(16),
    eug_r3_passwd character varying(16),
    eug_r3_no_of_conn numeric(3,0),
    eug_db_no_of_conn numeric(3,0),
    eug_r3_no_of_retry numeric(3,0),
    eug_db_no_of_retry numeric(3,0),
    eug_transaction_auto_retry numeric(3,0),
    eug_conn_log character varying(1),
    eug_auto_correction_yesno character varying(1),
    eug_logsize numeric(3,0),
    eug_logfile_path character varying(40),
    eug_xml_exchange_path character varying(40),
    eug_conn_exist character varying(1)
);


ALTER TABLE public.ezc_user_groups OWNER TO postgres;

--
-- TOC entry 283 (class 1259 OID 18673)
-- Name: ezc_user_product_favorites; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_user_product_favorites (
    epf_user_id character varying(20),
    epf_sys_key character varying(18),
    epf_favourite_group character varying(18),
    epf_mm_id numeric(18,0),
    epf_catalog_no numeric(8,0),
    epf_mat_no character varying(18),
    epf_product_sequence numeric(10,0),
    epf_type character varying(5),
    epf_itemcat character varying(32)
);


ALTER TABLE public.ezc_user_product_favorites OWNER TO postgres;

--
-- TOC entry 284 (class 1259 OID 18676)
-- Name: ezc_user_product_group_fav; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_user_product_group_fav (
    epgf_user_id character varying(20),
    epgf_sys_no numeric(3,0),
    epgf_product_group character varying(18),
    epgf_product_group_sequence numeric(10,0)
);


ALTER TABLE public.ezc_user_product_group_fav OWNER TO postgres;

--
-- TOC entry 285 (class 1259 OID 18679)
-- Name: ezc_user_roles; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_user_roles (
    eur_role_nr character varying(16),
    eur_role_type character varying(1),
    eur_language character varying(2),
    eur_role_description character varying(30),
    eur_deleted_flag character varying(1),
    eur_component character varying(20),
    eur_bus_domain character varying(20)
);


ALTER TABLE public.ezc_user_roles OWNER TO postgres;

--
-- TOC entry 286 (class 1259 OID 18682)
-- Name: ezc_users; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_users (
    eu_id character varying(20),
    eu_deletion_flag character varying(1),
    eu_first_name character varying(60),
    eu_middle_initial character varying(20),
    eu_last_name character varying(60),
    eu_email character varying(50),
    eu_created_date character varying(10),
    eu_changed_date character varying(10),
    eu_changed_by character varying(20),
    eu_valid_to_date character varying(10),
    eu_last_login_time character varying(20),
    eu_last_login_date character varying(10),
    eu_password character varying(512),
    eu_type numeric(1,0),
    eug_id numeric(3,0),
    eu_business_partner character varying(10),
    eu_is_built_in_user character varying(1),
    eu_current_number numeric(10,0),
    eu_telephone character varying(15),
    eu_mobile character varying(15),
    eu_fax character varying(100),
    eu_telephone_ext character varying(15),
    eu_otp character varying(10),
    eu_otp_valid timestamp without time zone
);


ALTER TABLE public.ezc_users OWNER TO postgres;

--
-- TOC entry 287 (class 1259 OID 18687)
-- Name: ezc_value_mapping; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_value_mapping (
    map_type character varying(20),
    value1 character varying(50),
    value2 character varying(500)
);


ALTER TABLE public.ezc_value_mapping OWNER TO postgres;

--
-- TOC entry 288 (class 1259 OID 18692)
-- Name: ezc_warranty_claims_comments; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_warranty_claims_comments (
    ewcc_doc_id character varying(20),
    ewcc_comments character varying(3000),
    ewcc_user_id character varying(100),
    ewcc_date timestamp without time zone NOT NULL,
    ewcc_ext1 character varying(500),
    ewcc_ext2 character varying(50),
    ewcc_visibility character varying(50)
);


ALTER TABLE public.ezc_warranty_claims_comments OWNER TO postgres;

--
-- TOC entry 289 (class 1259 OID 18697)
-- Name: ezc_warranty_claims_header; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_warranty_claims_header (
    ewch_uuid character varying(100),
    ewch_serial_no character varying(100),
    ewch_unit_installed_date character varying(100),
    ewch_failed_date character varying(100),
    ewch_customer_po character varying(100),
    ewch_first_name character varying(100),
    ewch_last_name character varying(100),
    ewch_addr_line_1 character varying(200),
    ewch_addr_line_2 character varying(200),
    ewch_email character varying(200),
    ewch_phone_no character varying(40),
    ewch_city character varying(100),
    ewch_state character varying(100),
    ewch_country character varying(100),
    ewch_comments character varying(400),
    ewch_sell_sold_to character varying(200),
    ewch_created_by character varying(100),
    ewch_created_on timestamp without time zone,
    ewch_ship_name character varying(200),
    ewch_ship_addr character varying(400),
    ewch_ship_city character varying(200),
    ewch_ship_state character varying(200),
    ewch_ship_zip_code character varying(20),
    ewch_ship_country character varying(200),
    ewch_status character varying(200),
    ewch_claim_no character varying(200),
    ewch_ship_to character varying(100),
    ewch_sales_org character varying(100),
    ewch_modified_by character varying(100),
    ewch_modified_on timestamp without time zone,
    ewch_ship_phone character varying(50),
    ewch_equip_name character varying(100),
    ewch_unit_name character varying(100),
    ewch_sold_to_name character varying(200),
    ewch_zip character varying(50),
    ewch_ext1 character varying(50),
    ewch_ext2 character varying(50),
    ewch_ext3 character varying(50)
);


ALTER TABLE public.ezc_warranty_claims_header OWNER TO postgres;

--
-- TOC entry 290 (class 1259 OID 18702)
-- Name: ezc_warranty_claims_items; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_warranty_claims_items (
    ewci_uuid character varying(100),
    ewci_item_type character varying(10),
    ewci_material character varying(100),
    ewci_quantity character varying(100),
    ewci_defect_code character varying(100),
    ewci_description character varying(100),
    ewci_labour_key character varying(100),
    ewci_comments character varying(1000),
    ewci_resolution character varying(200),
    ewci_status character varying(200)
);


ALTER TABLE public.ezc_warranty_claims_items OWNER TO postgres;

--
-- TOC entry 291 (class 1259 OID 18707)
-- Name: ezc_web_stats; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_web_stats (
    ews_user_id character varying(20),
    ews_syskey character varying(18),
    ews_sold_to character varying(18),
    ews_ip character varying(15),
    ews_logged_in timestamp without time zone,
    ews_logged_out timestamp without time zone
);


ALTER TABLE public.ezc_web_stats OWNER TO postgres;

--
-- TOC entry 292 (class 1259 OID 18752)
-- Name: ezc_wf_orgonagram; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_wf_orgonagram (
    ewo_code numeric(6,0),
    ewo_syskey character varying(18),
    ewo_lang character varying(2),
    ewo_template numeric(6,0),
    ewo_description character varying(100)
);


ALTER TABLE public.ezc_wf_orgonagram OWNER TO postgres;

--
-- TOC entry 293 (class 1259 OID 18755)
-- Name: ezc_wf_orgonagram_details; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ezc_wf_orgonagram_details (
    ewod_code numeric(6,0),
    ewod_level numeric(3,0),
    ewod_participant_type character varying(1),
    ewod_participant character varying(20),
    ewod_lang character varying(2),
    ewod_description character varying(100),
    ewod_parent character varying(20)
);


ALTER TABLE public.ezc_wf_orgonagram_details OWNER TO postgres;

--
-- TOC entry 298 (class 1259 OID 29134)
-- Name: schema_embeddings; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.schema_embeddings (
    id integer NOT NULL,
    table_name text NOT NULL,
    column_name text,
    description text NOT NULL,
    embedding json NOT NULL
);


ALTER TABLE public.schema_embeddings OWNER TO postgres;

--
-- TOC entry 297 (class 1259 OID 29133)
-- Name: schema_embeddings_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE public.schema_embeddings_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE public.schema_embeddings_id_seq OWNER TO postgres;

--
-- TOC entry 5256 (class 0 OID 0)
-- Dependencies: 297
-- Name: schema_embeddings_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE public.schema_embeddings_id_seq OWNED BY public.schema_embeddings.id;


--
-- TOC entry 5012 (class 2604 OID 29123)
-- Name: chat_messages id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.chat_messages ALTER COLUMN id SET DEFAULT nextval('public.chat_messages_id_seq'::regclass);


--
-- TOC entry 5013 (class 2604 OID 29137)
-- Name: schema_embeddings id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.schema_embeddings ALTER COLUMN id SET DEFAULT nextval('public.schema_embeddings_id_seq'::regclass);


--
-- TOC entry 5017 (class 2606 OID 29127)
-- Name: chat_messages chat_messages_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.chat_messages
    ADD CONSTRAINT chat_messages_pkey PRIMARY KEY (id);


--
-- TOC entry 5015 (class 2606 OID 29118)
-- Name: chat_sessions chat_sessions_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.chat_sessions
    ADD CONSTRAINT chat_sessions_pkey PRIMARY KEY (id);


--
-- TOC entry 5020 (class 2606 OID 29141)
-- Name: schema_embeddings schema_embeddings_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.schema_embeddings
    ADD CONSTRAINT schema_embeddings_pkey PRIMARY KEY (id);


--
-- TOC entry 5018 (class 1259 OID 29142)
-- Name: ix_schema_embeddings_table_name; Type: INDEX; Schema: public; Owner: postgres
--

CREATE INDEX ix_schema_embeddings_table_name ON public.schema_embeddings USING btree (table_name);


--
-- TOC entry 5021 (class 2606 OID 29128)
-- Name: chat_messages chat_messages_session_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.chat_messages
    ADD CONSTRAINT chat_messages_session_id_fkey FOREIGN KEY (session_id) REFERENCES public.chat_sessions(id);


-- Completed on 2025-12-29 17:08:18

--
-- PostgreSQL database dump complete
--

