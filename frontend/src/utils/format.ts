export const number = (value: number, maximumFractionDigits = 0) => new Intl.NumberFormat("es", { maximumFractionDigits }).format(value);
export const brl = (value: number) => new Intl.NumberFormat("es", { style: "currency", currency: "BRL", maximumFractionDigits: 2 }).format(value);
export const percent = (value: number) => new Intl.NumberFormat("es", { style: "percent", maximumFractionDigits: 1 }).format(value);
export const date = (value: string) => new Intl.DateTimeFormat("es", { day: "2-digit", month: "short", year: "numeric" }).format(new Date(`${value}T12:00:00`));
