import React from 'react';

export default function ProductCatalog({ products }) {
  return (
    <div className="catalog">
      <div className="catalog-header">📦 Product Catalog</div>
      <div className="catalog-list">
        {products.map(p => (
          <div key={p.id} className="product-card">
            <span className="product-icon">{p.image}</span>
            <div className="product-info">
              <strong>{p.name}</strong>
              <span className="product-price">${p.price}</span>
              <span className="product-desc">{p.description}</span>
              <span className="product-stock">{p.stock} in stock</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
