namespace DumbML {
    public static class Loss {
        public static Operation MSE(Operation labels, Operation truth) {
            Operation op = new Subtract(labels, truth);
            op = new Square(op);
            op = new ReduceSum(op, (int[])null);
            return op;
        }
    }
}
